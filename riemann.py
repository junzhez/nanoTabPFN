"""Riemann (binned) distribution utilities for regression PFNs.

Implements the discretized continuous distribution from Section 4 of
Müller et al., "Transformers Can Do Bayesian Inference" (ICLR 2022, arxiv 2112.10510):
y is partitioned into K buckets with equal prior probability 1/K, each bucket is
uniform within its boundaries, and the regression head outputs K logits per query.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_equiprobable_boundaries(samples: Tensor, num_buckets: int) -> Tensor:
    """Produce K+1 boundaries that split prior samples into K equiprobable buckets.

    Args:
        samples: 1D tensor of prior y samples (the more, the more stable).
        num_buckets: number of buckets K.
    Returns:
        Tensor of shape (num_buckets+1,), strictly monotonic.
    """
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    n = samples.numel()
    sorted_samples, _ = torch.sort(samples)
    # interior cuts at quantiles k/K for k=1..K-1
    qs_interior = torch.linspace(1.0 / num_buckets, 1.0 - 1.0 / num_buckets, num_buckets - 1, dtype=samples.dtype)
    interior = torch.quantile(sorted_samples, qs_interior)
    # outermost anchors at 0.5/n and 1 - 0.5/n (avoids the empirical min/max being too tight)
    lo = torch.quantile(sorted_samples, torch.tensor(0.5 / n, dtype=samples.dtype))
    hi = torch.quantile(sorted_samples, torch.tensor(1.0 - 0.5 / n, dtype=samples.dtype))
    boundaries = torch.cat([lo.reshape(1), interior, hi.reshape(1)])
    # ensure strictly monotonic
    eps = 1e-6
    for i in range(1, boundaries.numel()):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + eps
    return boundaries


class RiemannDistribution:
    """Equiprobable-boundary discretized distribution with uniform density per bucket.

    Each bucket [b_k, b_{k+1}] gets prior probability 1/K. Predicted density at y is
    p_k / (b_{k+1} - b_k), where p_k is softmax(logits)_k and k = bucket(y).
    """

    def __init__(self, boundaries: Tensor):
        assert boundaries.ndim == 1 and boundaries.numel() >= 3
        self.boundaries = boundaries.detach().clone()
        self.num_buckets = boundaries.numel() - 1
        self.widths = self.boundaries[1:] - self.boundaries[:-1]  # (K,)
        self.centers = 0.5 * (self.boundaries[1:] + self.boundaries[:-1])  # (K,)
        self.cum_widths = torch.cat([torch.zeros(1, dtype=self.widths.dtype), self.widths.cumsum(0)])  # (K+1,)

    def to(self, device) -> "RiemannDistribution":
        new = RiemannDistribution.__new__(RiemannDistribution)
        new.boundaries = self.boundaries.to(device)
        new.num_buckets = self.num_buckets
        new.widths = self.widths.to(device)
        new.centers = self.centers.to(device)
        new.cum_widths = self.cum_widths.to(device)
        return new

    def bucketize(self, y: Tensor) -> Tensor:
        """Map continuous y values to bucket indices in [0, K-1]."""
        # interior cut points are boundaries[1:-1]; bucketize maps to [0, K]
        idx = torch.bucketize(y.contiguous(), self.boundaries[1:-1].to(y.device))
        return idx.clamp(0, self.num_buckets - 1)

    def mean(self, logits: Tensor) -> Tensor:
        """E[y] under softmax(logits) per row.

        Args:
            logits: (..., K)
        Returns:
            (...)
        """
        probs = torch.softmax(logits, dim=-1)
        centers = self.centers.to(logits.device)
        return (probs * centers).sum(dim=-1)

    def quantile(self, logits: Tensor, q: float) -> Tensor:
        """Inverse CDF at q under the piecewise-uniform distribution defined by softmax(logits).

        Args:
            logits: (..., K)
            q: target quantile in (0, 1)
        Returns:
            (...) — tensor of y values with the same leading shape as logits.
        """
        probs = torch.softmax(logits, dim=-1)  # (..., K)
        cum = probs.cumsum(dim=-1)  # (..., K)
        # Find the smallest k such that cum[..., k] >= q.
        q_t = torch.full_like(cum[..., :1], q)
        ge = cum >= q_t  # (..., K)
        # First True per row; if none (numerical), use last bucket.
        first_idx = torch.where(ge.any(dim=-1, keepdim=True), ge.float().argmax(dim=-1, keepdim=True), torch.full_like(cum[..., :1], self.num_buckets - 1, dtype=torch.long))
        # cum_below = cumulative prob just below this bucket
        cum_below = torch.zeros_like(probs).scatter(-1, torch.zeros_like(first_idx), 0.0)
        # gather
        k = first_idx.squeeze(-1)  # (...)
        # cum mass strictly below bucket k:
        cum_left = torch.where(k > 0, cum.gather(-1, (k - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1), torch.zeros_like(k, dtype=probs.dtype))
        p_k = probs.gather(-1, k.unsqueeze(-1)).squeeze(-1).clamp(min=1e-30)
        b_lo = self.boundaries[:-1].to(logits.device).gather(0, k.reshape(-1)).reshape(k.shape)
        b_hi = self.boundaries[1:].to(logits.device).gather(0, k.reshape(-1)).reshape(k.shape)
        # within-bucket fraction
        frac = ((q - cum_left) / p_k).clamp(0.0, 1.0)
        return b_lo + frac * (b_hi - b_lo)

    def predict_density(self, logits: Tensor, y_grid: Tensor) -> Tensor:
        """Density at each y in y_grid for each row of logits.

        Args:
            logits: (..., K)
            y_grid: (G,)
        Returns:
            (..., G) — density values.
        """
        probs = torch.softmax(logits, dim=-1)  # (..., K)
        widths = self.widths.to(logits.device)
        density_per_bucket = probs / widths  # (..., K), uniform within each bucket
        idx = self.bucketize(y_grid)  # (G,)
        # density[..., g] = density_per_bucket[..., idx[g]]
        out = density_per_bucket.index_select(-1, idx.to(logits.device))
        # Zero out points outside [boundaries[0], boundaries[-1]]
        in_range = (y_grid >= self.boundaries[0].to(y_grid.device)) & (y_grid <= self.boundaries[-1].to(y_grid.device))
        out = out * in_range.to(out.dtype)
        return out
