"""Conditional Flow Matching model, objective, and ODE samplers."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class TimeConditionedVectorField(nn.Module):
    """Small MLP mapping ``(time, 2-D position)`` to a 2-D velocity."""

    def __init__(self, hidden_dim: int = 128, n_hidden_layers: int = 3) -> None:
        super().__init__()
        if hidden_dim <= 0 or n_hidden_layers <= 0:
            raise ValueError("hidden_dim and n_hidden_layers must be positive")

        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, time: Tensor, position: Tensor) -> Tensor:
        if position.ndim != 2 or position.shape[1] != 2:
            raise ValueError(f"expected position shape (batch, 2), got {tuple(position.shape)}")
        if time.ndim == 1:
            time = time[:, None]
        if time.shape != (position.shape[0], 1):
            raise ValueError(f"expected time shape ({position.shape[0]}, 1), got {tuple(time.shape)}")
        return self.network(torch.cat((time, position), dim=1))


def make_ot_cfm_training_pair(
    x_data: Tensor,
    *,
    sigma_min: float,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Construct a conditional optimal-transport Flow Matching batch.

    The affine conditional path is

        x_t = [1 - (1 - sigma_min)t] z + t x_data,

    where z is standard Gaussian noise. Differentiating the path gives the
    regression target ``x_data - (1 - sigma_min)z``. Optional ``noise`` and
    ``time`` arguments make the central equations exactly unit-testable.
    """
    if x_data.ndim != 2 or x_data.shape[1] != 2:
        raise ValueError(f"expected x_data shape (batch, 2), got {tuple(x_data.shape)}")
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError("sigma_min must lie in [0, 1)")

    batch_size = x_data.shape[0]
    if noise is None:
        noise = torch.randn(
            x_data.shape, generator=generator, device=x_data.device, dtype=x_data.dtype
        )
    if time is None:
        time = torch.rand(
            (batch_size, 1), generator=generator, device=x_data.device, dtype=x_data.dtype
        )
    if noise.shape != x_data.shape:
        raise ValueError(f"expected noise shape {tuple(x_data.shape)}, got {tuple(noise.shape)}")
    if time.shape != (batch_size, 1):
        raise ValueError(f"expected time shape ({batch_size}, 1), got {tuple(time.shape)}")

    endpoint_scale = 1.0 - sigma_min
    x_time = (1.0 - endpoint_scale * time) * noise + time * x_data
    target_velocity = x_data - endpoint_scale * noise
    return time, x_time, target_velocity


def conditional_flow_matching_loss(
    model: nn.Module,
    x_data: Tensor,
    *,
    sigma_min: float,
    generator: torch.Generator,
) -> Tensor:
    """Mean squared velocity-regression loss for one minibatch."""
    time, x_time, target_velocity = make_ot_cfm_training_pair(
        x_data, sigma_min=sigma_min, generator=generator
    )
    predicted_velocity = model(time, x_time)
    return (predicted_velocity - target_velocity).square().sum(dim=1).mean()


@torch.inference_mode()
def sample_midpoint(
    model: nn.Module,
    initial_noise: Tensor,
    *,
    n_function_evaluations: int,
    end_time: float = 1.0,
) -> Tensor:
    """Integrate dx/dt = v(t, x) with the explicit midpoint method.

    A midpoint step evaluates the vector field twice, so NFE must be positive
    and even. ``end_time`` is useful for inspecting intermediate transport.
    """
    if n_function_evaluations <= 0 or n_function_evaluations % 2:
        raise ValueError("n_function_evaluations must be a positive even integer")
    if not 0.0 <= end_time <= 1.0:
        raise ValueError("end_time must lie in [0, 1]")

    x = initial_noise.clone()
    n_steps = n_function_evaluations // 2
    step_size = end_time / n_steps
    for step in range(n_steps):
        t = step * step_size
        time = torch.full((len(x), 1), t, device=x.device, dtype=x.dtype)
        first_velocity = model(time, x)
        midpoint_time = time + 0.5 * step_size
        midpoint = x + 0.5 * step_size * first_velocity
        x = x + step_size * model(midpoint_time, midpoint)
    return x


@torch.inference_mode()
def sample_trajectory(
    model: nn.Module,
    initial_noise: Tensor,
    *,
    snapshot_times: Iterable[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    n_steps: int = 30,
) -> dict[float, Tensor]:
    """Integrate one trajectory and retain samples at selected grid times."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    requested = tuple(float(t) for t in snapshot_times)
    if any(t < 0.0 or t > 1.0 for t in requested):
        raise ValueError("snapshot times must lie in [0, 1]")

    grid_indices = {round(t * n_steps): t for t in requested}
    if any(not math.isclose(index / n_steps, t, abs_tol=1e-8) for index, t in grid_indices.items()):
        raise ValueError("every snapshot time must fall on the integration grid")

    x = initial_noise.clone()
    snapshots: dict[float, Tensor] = {}
    if 0 in grid_indices:
        snapshots[grid_indices[0]] = x.clone()

    step_size = 1.0 / n_steps
    for step in range(n_steps):
        time = torch.full((len(x), 1), step * step_size, device=x.device, dtype=x.dtype)
        first_velocity = model(time, x)
        midpoint_time = time + 0.5 * step_size
        midpoint = x + 0.5 * step_size * first_velocity
        x = x + step_size * model(midpoint_time, midpoint)
        completed = step + 1
        if completed in grid_indices:
            snapshots[grid_indices[completed]] = x.clone()
    return snapshots
