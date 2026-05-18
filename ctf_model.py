"""Counterfactual PFN inference wrappers.

`NanoTabPFNCtfRegressor` mirrors `NanoTabPFNRegressor` (model.py:198) but
accepts (Z, X) factual training data and (Z, X, X') counterfactual queries at
predict time. The model itself is an unmodified `NanoTabPFNModel`; we just
construct features as `[Z, X, X']` per row, with X'=X on training rows.

`CtfPFNFamily` holds K independently trained models and exposes per-model
predictions plus aggregate spread statistics that estimate the partial-
identification region for the counterfactual.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from model import NanoTabPFNModel


class NanoTabPFNCtfRegressor:
    """Counterfactual regressor wrapper.

    fit(Z_train, X_train, Y_train) stores factual training data; predict_*
    methods take (Z_test, X_test, X_intervention) and return P(Y_{X'} | Z, X).
    """

    def __init__(self, model: NanoTabPFNModel, riemann, device: torch.device, d_z: int):
        self.model = model.to(device)
        self.device = device
        self.riemann = riemann.to(device)
        self.d_z = d_z

    def fit(self, Z_train: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray):
        """Store factual training data. X' for training rows is set to X."""
        Z = np.asarray(Z_train, dtype=np.float32)
        X = np.asarray(X_train, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y_train, dtype=np.float32).reshape(-1)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if Z.shape[1] != self.d_z:
            raise ValueError(f"Z_train has {Z.shape[1]} cols; expected d_z={self.d_z}")
        if not (Z.shape[0] == X.shape[0] == Y.shape[0]):
            raise ValueError("Z_train, X_train, Y_train must have the same length.")
        self.Z_train = Z
        self.X_train = X
        self.Y_train = Y

    def _build_features(
        self,
        Z_test: np.ndarray,
        X_test: np.ndarray,
        X_intervention: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct (x, y_train) tensors for a forward pass.

        x: (1, N_train + N_test, d_z + 2)  with columns [Z, X, X'].
            train rows: X'=X (factual).
            test rows: X'=X_intervention (potentially counterfactual).
        y_train: (1, N_train) factual training targets.
        """
        Z_test = np.asarray(Z_test, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32).reshape(-1)
        X_intervention = np.asarray(X_intervention, dtype=np.float32).reshape(-1)
        if Z_test.ndim == 1:
            Z_test = Z_test.reshape(-1, 1)
        if Z_test.shape[1] != self.d_z:
            raise ValueError(f"Z_test has {Z_test.shape[1]} cols; expected d_z={self.d_z}")
        if not (Z_test.shape[0] == X_test.shape[0] == X_intervention.shape[0]):
            raise ValueError("Z_test, X_test, X_intervention must have the same length.")

        # Train rows: factual (X' = X).
        train_block = np.concatenate(
            [self.Z_train, self.X_train.reshape(-1, 1), self.X_train.reshape(-1, 1)],
            axis=1,
        )
        # Test rows: arbitrary intervention.
        test_block = np.concatenate(
            [Z_test, X_test.reshape(-1, 1), X_intervention.reshape(-1, 1)],
            axis=1,
        )
        x_full = np.concatenate([train_block, test_block], axis=0)  # (N, d_z+2)
        x_t = torch.from_numpy(x_full).unsqueeze(0).to(self.device)
        y_t = torch.from_numpy(self.Y_train).unsqueeze(0).to(self.device)
        return x_t, y_t

    def _logits(
        self, Z_test: np.ndarray, X_test: np.ndarray, X_intervention: np.ndarray
    ) -> torch.Tensor:
        x_t, y_t = self._build_features(Z_test, X_test, X_intervention)
        with torch.no_grad():
            out = self.model((x_t, y_t), train_test_split_index=len(self.X_train)).squeeze(0)
        return out  # (N_test, K)

    def predict_distribution(
        self, Z_test: np.ndarray, X_test: np.ndarray, X_intervention: np.ndarray
    ) -> np.ndarray:
        """Softmax PMF over Riemann buckets, shape (N_test, K)."""
        logits = self._logits(Z_test, X_test, X_intervention)
        return F.softmax(logits, dim=-1).cpu().numpy()

    def predict_density(
        self,
        Z_test: np.ndarray,
        X_test: np.ndarray,
        X_intervention: np.ndarray,
        y_grid: np.ndarray,
    ) -> np.ndarray:
        """Density at each y in y_grid, shape (N_test, len(y_grid))."""
        logits = self._logits(Z_test, X_test, X_intervention)
        y_grid_t = torch.as_tensor(y_grid, dtype=torch.float32, device=self.device)
        return self.riemann.predict_density(logits, y_grid_t).cpu().numpy()

    def predict_counterfactual(
        self,
        Z_test: np.ndarray,
        X_test: np.ndarray,
        X_intervention: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior mean and (alpha/2, 1-alpha/2) credible interval of Y_{X'}.

        Returns (mean, lo, hi) numpy arrays of shape (N_test,).
        """
        logits = self._logits(Z_test, X_test, X_intervention)
        mean = self.riemann.mean(logits)
        lo = self.riemann.quantile(logits, alpha / 2)
        hi = self.riemann.quantile(logits, 1 - alpha / 2)
        return mean.cpu().numpy(), lo.cpu().numpy(), hi.cpu().numpy()


class CtfPFNFamily:
    """A family of K independently trained CtfPFNs.

    Each member fits the observational data well; their disagreement on
    counterfactual queries (X' != X) is the empirical estimate of the partial
    identification region.
    """

    def __init__(
        self, models: list[NanoTabPFNModel], riemann, device: torch.device, d_z: int
    ):
        self.regressors = [
            NanoTabPFNCtfRegressor(m, riemann, device, d_z=d_z) for m in models
        ]
        self.K = len(models)

    def fit(self, Z_train: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray):
        for r in self.regressors:
            r.fit(Z_train, X_train, Y_train)

    def predict_counterfactual_family(
        self,
        Z_test: np.ndarray,
        X_test: np.ndarray,
        X_intervention: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """Per-member counterfactual predictions plus aggregate stats.

        Returns a dict with:
            means:        (K, N_test)  per-model posterior means
            los:          (K, N_test)  per-model alpha/2 quantiles
            his:          (K, N_test)  per-model 1-alpha/2 quantiles
            ensemble_mean:(N_test,)    mean over K
            family_std:   (N_test,)    std of means over K (ID-region width estimate)
            envelope_lo:  (N_test,)    min over K of los
            envelope_hi:  (N_test,)    max over K of his
        """
        means_list = []
        los_list = []
        his_list = []
        for r in self.regressors:
            mean, lo, hi = r.predict_counterfactual(Z_test, X_test, X_intervention, alpha=alpha)
            means_list.append(mean)
            los_list.append(lo)
            his_list.append(hi)
        means = np.stack(means_list, axis=0)
        los = np.stack(los_list, axis=0)
        his = np.stack(his_list, axis=0)
        return {
            "means": means,
            "los": los,
            "his": his,
            "ensemble_mean": means.mean(axis=0),
            "family_std": means.std(axis=0),
            "envelope_lo": los.min(axis=0),
            "envelope_hi": his.max(axis=0),
        }
