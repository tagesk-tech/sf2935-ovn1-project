"""Two-dimensional checkerboard data."""

import numpy as np
import torch


def checkerboard(n, seed, device="cpu"):
    rng = np.random.default_rng(seed)
    x = 4 * rng.random(n) - 2
    y = rng.random(n) - 2 * rng.integers(0, 2, n)
    y += np.mod(np.floor(x), 2)
    points = 2 * np.column_stack((x, y))
    return torch.tensor(points, dtype=torch.float32, device=device)


def minibatch(data, size, generator):
    indices = torch.randint(len(data), (size,), generator=generator, device=data.device)
    return data[indices]
