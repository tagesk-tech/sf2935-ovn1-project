"""Data utilities for the two-dimensional checkerboard experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CheckerboardConfig:
    """Geometry of the synthetic checkerboard distribution."""

    x_scale: float = 2.0
    y_scale: float = 2.0


def sample_checkerboard(
    n_samples: int,
    *,
    seed: int,
    config: CheckerboardConfig = CheckerboardConfig(),
    device: torch.device | str = "cpu",
) -> Tensor:
    """Draw deterministic samples from an eight-tile checkerboard.

    A local NumPy generator is used so this function does not change global
    random state. The result has shape ``(n_samples, 2)`` and dtype float32.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    x = 4.0 * rng.random(n_samples) - 2.0
    y = rng.random(n_samples) - 2.0 * rng.integers(0, 2, size=n_samples)
    y = y + np.mod(np.floor(x), 2.0)
    samples = np.column_stack((config.x_scale * x, config.y_scale * y))
    return torch.as_tensor(samples, dtype=torch.float32, device=device)


def train_test_checkerboard(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Create independent, reproducible training and held-out samples."""
    train = sample_checkerboard(n_train, seed=seed, device=device)
    test = sample_checkerboard(n_test, seed=seed + 1, device=device)
    return train, test


def sample_minibatch(data: Tensor, batch_size: int, generator: torch.Generator) -> Tensor:
    """Sample a minibatch with replacement from an in-memory dataset."""
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"expected data with shape (n, 2), got {tuple(data.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = torch.randint(len(data), (batch_size,), generator=generator, device=data.device)
    return data[indices]
