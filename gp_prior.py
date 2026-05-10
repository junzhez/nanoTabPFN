"""On-the-fly Gaussian-Process regression prior for PFN training.

Mirrors the output dict shape of `train.PriorDumpDataLoader` so the existing
`train()` loop works unchanged: yields `{"x": (B, N, F), "y": (B, N), "train_test_split_index": int}`
but with float-valued y (samples from a zero-mean GP with a fixed RBF kernel).

Also provides a closed-form GP posterior used as the exact baseline for Figure 3
of arxiv 2112.10510, Section 5.1.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rbf_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    """Batched RBF kernel.

    Args:
        x1: (..., N, F)
        x2: (..., M, F)
    Returns:
        (..., N, M)
    """
    # Squared distances: ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1.x2
    sq1 = (x1 ** 2).sum(dim=-1, keepdim=True)         # (..., N, 1)
    sq2 = (x2 ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (..., 1, M)
    cross = x1 @ x2.transpose(-1, -2)                  # (..., N, M)
    sqdist = (sq1 + sq2 - 2 * cross).clamp_min(0.0)
    return variance * torch.exp(-0.5 * sqdist / (lengthscale ** 2))


class GPPriorDataLoader:
    """Generates batches of GP-prior datasets on the fly.

    Each yielded batch contains `batch_size` independent draws from a zero-mean GP
    with the given RBF kernel. `train_test_split_index` is sampled once per batch
    so the existing `train()` loop (which expects a single int) is unchanged.

    Args:
        num_steps: number of batches per "epoch".
        batch_size: B.
        n_per_dataset: total points per dataset (N = n_train + n_test).
        num_features: F. Use 1 for the Figure 3 1D setup.
        lengthscale: RBF length scale.
        variance: RBF signal variance.
        noise: jitter added to the diagonal of K for numerical stability.
        n_train_min, n_train_max: range to sample n_train uniformly per step.
        device: torch device.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        n_per_dataset: int = 100,
        num_features: int = 1,
        lengthscale: float = 0.1,
        variance: float = 1.0,
        noise: float = 1e-4,
        n_train_min: int = 2,
        n_train_max: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.n_per_dataset = n_per_dataset
        self.num_features = num_features
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.n_train_min = n_train_min
        self.n_train_max = n_train_max if n_train_max is not None else max(n_per_dataset - 5, n_train_min + 1)
        if device is None:
            device = "cpu"
        self.device = device

    def __iter__(self):
        gen = torch.Generator(device="cpu")  # CPU generator; tensors moved to device after.
        for _ in range(self.num_steps):
            B, N, F = self.batch_size, self.n_per_dataset, self.num_features
            x = torch.rand(B, N, F, generator=gen)  # U[0,1]
            K = rbf_kernel(x, x, self.lengthscale, self.variance) + self.noise * torch.eye(N).expand(B, N, N)
            # robust cholesky in float64
            K64 = K.to(torch.float64)
            try:
                L = torch.linalg.cholesky(K64)
            except RuntimeError:
                # fallback with more jitter
                L = torch.linalg.cholesky(K64 + 1e-4 * torch.eye(N, dtype=torch.float64).expand(B, N, N))
            eps = torch.randn(B, N, 1, generator=gen, dtype=torch.float64)
            y = (L @ eps).squeeze(-1).to(torch.float32)  # (B, N)
            n_train = int(torch.randint(self.n_train_min, self.n_train_max + 1, (1,), generator=gen).item())
            yield {
                "x": x.to(self.device),
                "y": y.to(self.device),
                "train_test_split_index": n_train,
            }

    def __len__(self):
        return self.num_steps


def sample_y_for_boundaries(
    num_samples: int = 50000,
    n_per_dataset: int = 100,
    num_features: int = 1,
    lengthscale: float = 0.1,
    variance: float = 1.0,
    noise: float = 1e-4,
    chunk_size: int = 256,
) -> Tensor:
    """Draw GP-prior y samples in bulk, for `make_equiprobable_boundaries`.

    Returns a 1D tensor of length ~num_samples (rounded up to multiples of n_per_dataset).
    """
    n_datasets = math.ceil(num_samples / n_per_dataset)
    out = []
    remaining = n_datasets
    while remaining > 0:
        B = min(chunk_size, remaining)
        N, F = n_per_dataset, num_features
        x = torch.rand(B, N, F)
        K = rbf_kernel(x, x, lengthscale, variance) + noise * torch.eye(N).expand(B, N, N)
        K64 = K.to(torch.float64)
        try:
            L = torch.linalg.cholesky(K64)
        except RuntimeError:
            L = torch.linalg.cholesky(K64 + 1e-4 * torch.eye(N, dtype=torch.float64).expand(B, N, N))
        eps = torch.randn(B, N, 1, dtype=torch.float64)
        y = (L @ eps).squeeze(-1).to(torch.float32)  # (B, N)
        out.append(y.reshape(-1))
        remaining -= B
    return torch.cat(out)[:num_samples]


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    lengthscale: float,
    variance: float,
    noise: float,
) -> tuple[Tensor, Tensor]:
    """Closed-form GP posterior given (x_train, y_train) and query points x_test.

    Args:
        x_train: (n, F)
        y_train: (n,)
        x_test:  (m, F)
    Returns:
        mean: (m,) posterior mean
        var:  (m,) posterior variance
    """
    x_train64 = x_train.to(torch.float64)
    y_train64 = y_train.to(torch.float64)
    x_test64 = x_test.to(torch.float64)
    n = x_train64.shape[0]
    K = rbf_kernel(x_train64, x_train64, lengthscale, variance) + noise * torch.eye(n, dtype=torch.float64)
    # add small jitter for stability
    jitter = 1e-6
    while True:
        try:
            L = torch.linalg.cholesky(K + jitter * torch.eye(n, dtype=torch.float64))
            break
        except RuntimeError:
            jitter *= 10
            if jitter > 1.0:
                raise
    K_s = rbf_kernel(x_test64, x_train64, lengthscale, variance)        # (m, n)
    K_ss = rbf_kernel(x_test64, x_test64, lengthscale, variance)        # (m, m)
    alpha = torch.cholesky_solve(y_train64.unsqueeze(-1), L).squeeze(-1)  # (n,)
    mean = K_s @ alpha                                                  # (m,)
    v = torch.linalg.solve_triangular(L, K_s.transpose(-1, -2), upper=False)  # (n, m)
    var = K_ss.diagonal() - (v ** 2).sum(dim=0)                          # (m,)
    var = var.clamp_min(0.0)
    return mean.to(torch.float32), var.to(torch.float32)
