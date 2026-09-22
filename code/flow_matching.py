"""Conditional Flow Matching and midpoint ODE sampling."""

import torch
from torch import nn


class VectorField(nn.Module):
    def __init__(self, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(3, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.net(torch.cat((t, x), dim=1))


def cfm_loss(model, x_data, sigma_min, generator):
    """The conditional OT path and velocity-regression objective."""
    z = torch.randn(x_data.shape, generator=generator, device=x_data.device)
    t = torch.rand((len(x_data), 1), generator=generator, device=x_data.device)
    x_t = (1 - (1 - sigma_min) * t) * z + t * x_data
    target = x_data - (1 - sigma_min) * z
    return ((model(t, x_t) - target) ** 2).sum(dim=1).mean()


@torch.inference_mode()
def sample(model, noise, nfe=20, end_time=1.0):
    """Solve dx/dt = v(t,x) using two evaluations per midpoint step."""
    x = noise.clone()
    steps = nfe // 2
    dt = end_time / steps
    for step in range(steps):
        t = torch.full((len(x), 1), step * dt, device=x.device)
        velocity = model(t, x)
        midpoint = x + 0.5 * dt * velocity
        x += dt * model(t + 0.5 * dt, midpoint)
    return x
