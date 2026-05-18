"""Counterfactual training loop for CtfPFN.

Mirrors `train.train` but with two changes:
  1. The prior yields an extra `factual_mask: (B, N)` boolean field; rows where
     the mask is False (counterfactual queries with X' != X) have their bucketized
     target overwritten with -100 so PyTorch's `nn.CrossEntropyLoss` (default
     `ignore_index=-100`) skips them in the gradient.
  2. `train_ctf_family` runs the same training procedure K times with different
     random seeds, returning K independently trained models. The family's spread
     on counterfactual queries (X' != X) characterizes the partial-identification
     region that random-init alone exposes.
"""

from __future__ import annotations

import random
import time

import numpy as np
import schedulefree
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import NanoTabPFNModel


def set_randomness_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def train_ctf(
    model: NanoTabPFNModel,
    prior: DataLoader,
    riemann,
    lr: float = 1e-4,
    device: torch.device | None = None,
    steps_per_eval: int = 50,
    eval_func=None,
    verbose: bool = True,
):
    """Train a single CtfPFN.

    Args:
        model: NanoTabPFNModel with `num_outputs == riemann.num_buckets`.
        prior: iterable yielding {"x", "y", "train_test_split_index", "factual_mask"}.
        riemann: RiemannDistribution used to bucketize continuous y.
        lr: learning rate for AdamWScheduleFree.
        device: torch device; auto-detected if None.
        steps_per_eval: log every N steps.
        eval_func: optional callable taking (NanoTabPFNCtfRegressor) -> dict of metrics.
        verbose: whether to print per-step progress.

    Returns:
        (model, eval_history) where eval_history is a list of (train_time, dict).
    """
    if device is None:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    # ignore_index=-100 is the default; we set it explicitly for clarity.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    optimizer.train()

    train_time = 0.0
    eval_history = []
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            data = (
                full_data["x"].to(device),
                full_data["y"][:, :train_test_split_index].to(device),
            )
            targets = full_data["y"].to(device)
            factual_mask = full_data["factual_mask"].to(device)

            output = model(data, train_test_split_index=train_test_split_index)  # (B, N_test, K)

            # Slice to test rows and flatten.
            targets = targets[:, train_test_split_index:].reshape(-1)
            mask_flat = factual_mask[:, train_test_split_index:].reshape(-1)

            # Bucketize continuous y, then mask counterfactual rows out via -100.
            targets = riemann.bucketize(targets)
            targets = torch.where(mask_flat, targets, torch.full_like(targets, -100))

            output = output.reshape(-1, output.shape[-1])

            loss = criterion(output, targets)
            loss.backward()
            total_loss = float(loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            if step % steps_per_eval == steps_per_eval - 1:
                if eval_func is not None:
                    from ctf_model import NanoTabPFNCtfRegressor  # local import to avoid cycle
                    model.eval()
                    optimizer.eval()
                    d_z = full_data["x"].shape[-1] - 2
                    wrapper = NanoTabPFNCtfRegressor(model, riemann, device, d_z=d_z)
                    scores = eval_func(wrapper)
                    eval_history.append((train_time, scores))
                    if verbose:
                        score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
                        print(f"step {step:5d} | time {train_time:7.1f}s | loss {total_loss:7.4f} | {score_str}")
                    model.train()
                    optimizer.train()
                elif verbose:
                    factual_frac = float(mask_flat.float().mean().detach().cpu())
                    print(
                        f"step {step:5d} | time {train_time:7.1f}s | loss {total_loss:7.4f} | "
                        f"factual_frac {factual_frac:.2f}"
                    )
    except KeyboardInterrupt:
        pass

    model.eval()
    optimizer.eval()
    return model, eval_history


def train_ctf_family(
    K: int,
    prior_factory,
    model_factory,
    riemann,
    seeds: list[int] | None = None,
    lr: float = 1e-4,
    device: torch.device | None = None,
    steps_per_eval: int = 50,
    eval_func=None,
    verbose: bool = True,
):
    """Train K CtfPFNs from independent seeds.

    Each model gets its own seed, its own freshly built model (via
    `model_factory(seed)`), and its own freshly built prior data loader (via
    `prior_factory(seed)`). The family's disagreement on counterfactual queries
    is the empirical estimate of the partial-identification region.

    Args:
        K: number of family members.
        prior_factory: seed -> DataLoader-compatible iterable.
        model_factory: seed -> NanoTabPFNModel.
        riemann: RiemannDistribution shared across the family.
        seeds: optional explicit list of K seeds; defaults to range(K).
        lr, device, steps_per_eval, eval_func, verbose: passed through.

    Returns:
        list[NanoTabPFNModel] of length K.
    """
    if seeds is None:
        seeds = list(range(K))
    if len(seeds) != K:
        raise ValueError(f"Expected {K} seeds, got {len(seeds)}")
    if device is None:
        device = get_default_device()

    models: list[NanoTabPFNModel] = []
    for k, seed in enumerate(seeds):
        if verbose:
            print(f"=== Family member {k+1}/{K} (seed={seed}) ===")
        set_randomness_seed(seed)
        model = model_factory(seed)
        prior = prior_factory(seed)
        model, _ = train_ctf(
            model,
            prior,
            riemann=riemann,
            lr=lr,
            device=device,
            steps_per_eval=steps_per_eval,
            eval_func=eval_func,
            verbose=verbose,
        )
        models.append(model)
    return models
