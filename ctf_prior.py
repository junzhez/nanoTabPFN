"""On-the-fly counterfactual prior for CtfPFN training.

Generates synthetic datasets from a linear structural causal model (SCM) with
correlated Gaussian noises between treatment and outcome (i.e. an unobserved
confounder), bounded into [-1, 1] via tanh. The non-zero noise correlation is
what makes the counterfactual P(Y_{x'} | X=x, Z=z) only partially identified
from the observational distribution P(Y, X | Z).

Each yielded batch matches the existing prior format expected by the train loop
(`{"x": (B, N, F), "y": (B, N), "train_test_split_index": int}`) plus an extra
boolean `"factual_mask": (B, N)` tensor that flags rows where X' == X. Only
factual rows contribute gradient (consistency axiom).

Feature layout per row: `[Z_1, ..., Z_d, X, X']`, so F = d_z + 2.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class LinearSCMCtfPriorDataLoader:
    """Generates batches of linear-SCM counterfactual datasets on the fly.

    Per dataset (broadcast across rows):
        a_xz   ~ N(0, 0.7^2)  shape (d_z,)
        beta   ~ U(-1.5, 1.5) scalar (true causal effect of X on Y)
        gamma  ~ N(0, 0.5^2)  shape (d_z,)
        rho    ~ U(-rho_max, rho_max)  scalar (confounder correlation)
        sigma_x, sigma_y ~ U(sigma_min, sigma_max)
    Per row:
        Z      ~ N(0, I_{d_z})
        e_x = sigma_x * eps_1
        e_y = sigma_y * (rho * eps_1 + sqrt(1 - rho^2) * eps_2)
        X    = tanh(a_xz . Z + e_x)
        with prob ctf_fraction:        X' ~ U(-1, 1)   (counterfactual row)
        otherwise:                     X' = X          (factual row)
        Y    = tanh(beta * X' + gamma . Z + e_y)
    Critical invariant: the same e_y is used for both factual and counterfactual
    Y, so on counterfactual rows Y is the unit-level potential outcome the
    model never sees as a label.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        n_per_dataset: int = 200,
        d_z: int = 3,
        ctf_fraction: float = 0.5,
        n_train_min: int = 50,
        n_train_max: int | None = None,
        a_xz_std: float = 0.7,
        beta_range: float = 1.5,
        gamma_std: float = 0.5,
        rho_max: float = 0.9,
        sigma_min: float = 0.3,
        sigma_max: float = 1.0,
        device: torch.device | str | None = None,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.n_per_dataset = n_per_dataset
        self.d_z = d_z
        self.ctf_fraction = ctf_fraction
        self.n_train_min = n_train_min
        self.n_train_max = n_train_max if n_train_max is not None else max(n_per_dataset - 5, n_train_min + 1)
        self.a_xz_std = a_xz_std
        self.beta_range = beta_range
        self.gamma_std = gamma_std
        self.rho_max = rho_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device if device is not None else "cpu"

    @property
    def num_features(self) -> int:
        return self.d_z + 2

    def __iter__(self):
        gen = torch.Generator(device="cpu")
        B, N, d = self.batch_size, self.n_per_dataset, self.d_z
        for _ in range(self.num_steps):
            # Per-dataset SCM parameters (one per batch element, broadcast over N rows).
            a_xz = self.a_xz_std * torch.randn(B, 1, d, generator=gen)             # (B,1,d)
            beta = (2.0 * torch.rand(B, 1, generator=gen) - 1.0) * self.beta_range  # (B,1)
            gamma = self.gamma_std * torch.randn(B, 1, d, generator=gen)            # (B,1,d)
            rho = (2.0 * torch.rand(B, 1, generator=gen) - 1.0) * self.rho_max      # (B,1)
            sigma_x = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(B, 1, generator=gen)
            sigma_y = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(B, 1, generator=gen)

            # Per-row latents.
            Z = torch.randn(B, N, d, generator=gen)                                 # (B,N,d)
            eps_1 = torch.randn(B, N, generator=gen)
            eps_2 = torch.randn(B, N, generator=gen)
            e_x = sigma_x * eps_1
            e_y = sigma_y * (rho * eps_1 + torch.sqrt(torch.clamp(1.0 - rho ** 2, min=0.0)) * eps_2)

            X_raw = (Z * a_xz).sum(-1) + e_x                                        # (B,N)
            X = torch.tanh(X_raw)

            # Counterfactual mask: 1 == factual (X'=X), 0 == counterfactual.
            factual_mask = torch.rand(B, N, generator=gen) >= self.ctf_fraction
            X_alt = 2.0 * torch.rand(B, N, generator=gen) - 1.0                     # U[-1,1]
            X_prime = torch.where(factual_mask, X, X_alt)

            Y_raw = beta * X_prime + (Z * gamma).sum(-1) + e_y
            Y = torch.tanh(Y_raw)

            x = torch.cat([Z, X.unsqueeze(-1), X_prime.unsqueeze(-1)], dim=-1)      # (B,N,d+2)

            n_train = int(torch.randint(self.n_train_min, self.n_train_max + 1, (1,), generator=gen).item())
            yield {
                "x": x.to(self.device),
                "y": Y.to(self.device),
                "train_test_split_index": n_train,
                "factual_mask": factual_mask.to(self.device),
            }

    def __len__(self):
        return self.num_steps


def sample_y_for_boundaries(
    num_samples: int = 50000,
    n_per_dataset: int = 200,
    d_z: int = 3,
    ctf_fraction: float = 0.5,
    chunk_size: int = 256,
    **kwargs,
) -> Tensor:
    """Bulk sample of y under the prior, for `make_equiprobable_boundaries`.

    Note: with bounded outcomes (tanh) we usually prefer fixed uniform boundaries
    on [-1, 1] (see `uniform_boundaries`). This helper is provided for parity
    with `gp_prior.sample_y_for_boundaries` if you ever want equiprobable bins.
    """
    n_datasets = math.ceil(num_samples / n_per_dataset)
    out = []
    remaining = n_datasets
    while remaining > 0:
        B = min(chunk_size, remaining)
        loader = LinearSCMCtfPriorDataLoader(
            num_steps=1,
            batch_size=B,
            n_per_dataset=n_per_dataset,
            d_z=d_z,
            ctf_fraction=ctf_fraction,
            n_train_min=1,
            n_train_max=n_per_dataset - 1,
            **kwargs,
        )
        batch = next(iter(loader))
        out.append(batch["y"].reshape(-1).cpu())
        remaining -= B
    return torch.cat(out)[:num_samples]


def uniform_boundaries(num_buckets: int, lo: float = -1.0, hi: float = 1.0) -> Tensor:
    """K+1 equally spaced boundaries on [lo, hi] for use with RiemannDistribution.

    Matches the bounded-outcome support after tanh. Preferred over equiprobable
    boundaries for this prior because the equiprobable cuts pile up at the
    saturated tails near +/-1 and waste resolution.
    """
    return torch.linspace(lo, hi, num_buckets + 1)
