"""Train, evaluate, and plot the reduced Flow Matching reproduction.

Commands:
    python code/experiments.py --self-test
    python code/experiments.py --quick
    python code/experiments.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from data import sample_checkerboard, sample_minibatch, train_test_checkerboard
from flow_matching import (
    TimeConditionedVectorField,
    conditional_flow_matching_loss,
    make_ot_cfm_training_pair,
    sample_midpoint,
    sample_trajectory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (11, 22, 33)
    sigma_values: tuple[float, ...] = (0.01, 0.10, 0.30)
    baseline_sigma: float = 0.10
    n_train: int = 20_000
    n_test: int = 5_000
    n_generated: int = 5_000
    batch_size: int = 512
    train_steps: int = 5_000
    learning_rate: float = 2e-3
    hidden_dim: int = 128
    n_hidden_layers: int = 3
    metric_projections: int = 256
    evaluation_nfe: int = 20
    panel_nfes: tuple[int, ...] = (4, 8, 10, 20)


def set_seed(seed: int) -> None:
    """Set all random sources used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    config: ExperimentConfig,
    *,
    seed: int,
    sigma_min: float,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    """Train one neural vector field and return its loss history."""
    set_seed(seed)
    train_data, _ = train_test_checkerboard(
        config.n_train, config.n_test, seed=10_000 + seed, device=device
    )
    model = TimeConditionedVectorField(
        hidden_dim=config.hidden_dim, n_hidden_layers=config.n_hidden_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(20_000 + seed)
    losses: list[float] = []

    model.train()
    for step in range(config.train_steps):
        batch = sample_minibatch(train_data, config.batch_size, generator)
        loss = conditional_flow_matching_loss(
            model, batch, sigma_min=sigma_min, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(config.train_steps // 10, 1) == 0:
            print(
                f"seed={seed:02d} sigma={sigma_min:.3f} "
                f"step={step + 1:5d}/{config.train_steps} loss={losses[-1]:.5f}",
                flush=True,
            )
    return model.eval(), losses


@torch.inference_mode()
def sliced_wasserstein_distance(
    first: Tensor,
    second: Tensor,
    *,
    n_projections: int,
    seed: int,
) -> float:
    """Estimate 2-Wasserstein distance averaged over random 1-D slices."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("inputs must be matrices with the same feature dimension")
    n = min(len(first), len(second))
    first = first[:n]
    second = second[:n]
    generator = torch.Generator(device=first.device).manual_seed(seed)
    directions = torch.randn(
        (n_projections, first.shape[1]), generator=generator, device=first.device
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    first_projection = torch.sort(first @ directions.T, dim=0).values
    second_projection = torch.sort(second @ directions.T, dim=0).values
    return float(torch.sqrt((first_projection - second_projection).square().mean()).cpu())


def plot_scatter_panels(samples: list[Tensor], titles: list[str], destination: Path) -> None:
    """Save equally scaled scatter panels for qualitative comparison."""
    fig, axes = plt.subplots(
        1, len(samples), figsize=(3.0 * len(samples), 3.0), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, points, title in zip(axes, samples, titles, strict=True):
        values = points.detach().cpu().numpy()
        axis.scatter(values[:, 0], values[:, 1], s=2, alpha=0.45, rasterized=True)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlim(-5.0, 5.0)
        axis.set_ylim(-5.0, 5.0)
        axis.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(losses: list[float], destination: Path) -> None:
    """Plot a moving-average training loss."""
    window = min(100, max(1, len(losses) // 10))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, axis = plt.subplots(figsize=(5.5, 3.3))
    axis.plot(np.arange(window - 1, len(losses)), smoothed)
    axis.set(
        xlabel="Training step",
        ylabel="CFM loss",
        title=f"Training loss ({window}-step mean)",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_single(
    config: ExperimentConfig,
    *,
    seed: int,
    sigma_min: float,
    device: torch.device,
    results_dir: Path,
) -> tuple[dict[str, float | int], nn.Module, list[float]]:
    """Train and evaluate one seed/sigma configuration."""
    model, losses = train_model(config, seed=seed, sigma_min=sigma_min, device=device)
    test = sample_checkerboard(config.n_test, seed=30_000 + seed, device=device)
    generator = torch.Generator(device=device).manual_seed(40_000 + seed)
    noise = torch.randn((config.n_generated, 2), generator=generator, device=device)
    generated = sample_midpoint(model, noise, n_function_evaluations=config.evaluation_nfe)
    metric = sliced_wasserstein_distance(
        test,
        generated,
        n_projections=config.metric_projections,
        seed=50_000 + seed,
    )
    row: dict[str, float | int] = {
        "seed": seed,
        "sigma_min": sigma_min,
        "sliced_wasserstein": metric,
        "final_training_loss": float(np.mean(losses[-100:])),
    }
    checkpoint = results_dir / f"model_seed-{seed}_sigma-{sigma_min:.3f}.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config), **row}, checkpoint)
    return row, model, losses


def create_reproduction_figures(
    model: nn.Module,
    losses: list[float],
    config: ExperimentConfig,
    *,
    seed: int,
    device: torch.device,
    results_dir: Path,
) -> None:
    """Create the reduced Figure-4-style trajectory and NFE panels."""
    generator = torch.Generator(device=device).manual_seed(60_000 + seed)
    noise = torch.randn((2_500, 2), generator=generator, device=device)
    target = sample_checkerboard(len(noise), seed=70_000 + seed, device=device)
    snapshots = sample_trajectory(model, noise, n_steps=30)
    times = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    plot_scatter_panels(
        [snapshots[t] for t in times],
        [f"$t={t:.2f}$" for t in times],
        results_dir / "reproduction_trajectory.png",
    )
    generated_by_nfe = {
        nfe: sample_midpoint(model, noise, n_function_evaluations=nfe)
        for nfe in config.panel_nfes
    }
    plot_scatter_panels(
        [target, *generated_by_nfe.values()],
        ["Target data", *(f"NFE = {nfe}" for nfe in config.panel_nfes)],
        results_dir / "reproduction_nfe.png",
    )
    with (results_dir / "nfe_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["nfe", "sliced_wasserstein"], lineterminator="\n"
        )
        writer.writeheader()
        for nfe, generated in generated_by_nfe.items():
            writer.writerow(
                {
                    "nfe": nfe,
                    "sliced_wasserstein": sliced_wasserstein_distance(
                        target,
                        generated,
                        n_projections=config.metric_projections,
                        seed=80_000 + seed,
                    ),
                }
            )
    plot_training_curve(losses, results_dir / "baseline_training_loss.png")


def write_summary(rows: list[dict[str, float | int]], results_dir: Path) -> None:
    """Write raw metrics, grouped statistics, and the variation plot."""
    fieldnames = ["seed", "sigma_min", "sliced_wasserstein", "final_training_loss"]
    with (results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(row["sigma_min"]), []).append(float(row["sliced_wasserstein"]))
    summary = {
        str(sigma): {
            "mean_sliced_wasserstein": float(np.mean(values)),
            "std_sliced_wasserstein": (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            ),
            "n_seeds": len(values),
        }
        for sigma, values in grouped.items()
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    sigmas = sorted(grouped)
    means = [np.mean(grouped[sigma]) for sigma in sigmas]
    errors = [
        np.std(grouped[sigma], ddof=1) if len(grouped[sigma]) > 1 else 0.0 for sigma in sigmas
    ]
    fig, axis = plt.subplots(figsize=(5.5, 3.3))
    axis.errorbar(sigmas, means, yerr=errors, marker="o", capsize=4)
    axis.set(
        xlabel="$\\sigma_{min}$",
        ylabel="Sliced Wasserstein distance",
        title="Endpoint smoothing variation",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "variation_sigma_min.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def self_test() -> None:
    """Run deterministic checks before spending time on training."""
    first = sample_checkerboard(32, seed=7)
    second = sample_checkerboard(32, seed=7)
    assert first.shape == (32, 2)
    torch.testing.assert_close(first, second)

    model = TimeConditionedVectorField(hidden_dim=16, n_hidden_layers=1)
    assert model(torch.zeros(4, 1), torch.zeros(4, 2)).shape == (4, 2)

    x_data = torch.tensor([[3.0, 1.0]])
    noise = torch.tensor([[1.0, -2.0]])
    time = torch.tensor([[0.25]])
    _, x_time, target = make_ot_cfm_training_pair(
        x_data, sigma_min=0.1, noise=noise, time=time
    )
    torch.testing.assert_close(x_time, torch.tensor([[1.525, -1.3]]))
    torch.testing.assert_close(target, torch.tensor([[2.1, 2.8]]))

    class ConstantField(nn.Module):
        def forward(self, time: Tensor, position: Tensor) -> Tensor:
            return torch.ones_like(position)

    start = torch.zeros(5, 2)
    end = sample_midpoint(ConstantField(), start, n_function_evaluations=4)
    torch.testing.assert_close(end, torch.ones_like(start))

    identical_distance = sliced_wasserstein_distance(
        first, first.clone(), n_projections=16, seed=8
    )
    assert identical_distance == 0.0
    print("All self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke run, not report evidence")
    parser.add_argument("--self-test", action="store_true", help="run deterministic checks and exit")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--train-steps", type=int, help="override the configured training budget")
    parser.add_argument("--seeds", type=int, nargs="+", help="override the configured random seeds")
    parser.add_argument(
        "--sigma-values", type=float, nargs="+", help="override the sigma_min variation values"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    config = ExperimentConfig()
    if args.quick:
        config = replace(
            config,
            seeds=(11,),
            sigma_values=(config.baseline_sigma,),
            n_train=2_000,
            n_test=1_000,
            n_generated=1_000,
            train_steps=300,
            hidden_dim=64,
            n_hidden_layers=2,
            metric_projections=64,
        )
    if args.train_steps is not None:
        if args.train_steps <= 0:
            raise SystemExit("--train-steps must be positive")
        config = replace(config, train_steps=args.train_steps)
    if args.seeds is not None:
        config = replace(config, seeds=tuple(args.seeds))
    if args.sigma_values is not None:
        if any(not 0.0 <= value < 1.0 for value in args.sigma_values):
            raise SystemExit("every --sigma-values entry must lie in [0, 1)")
        config = replace(config, sigma_values=tuple(args.sigma_values))

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "device": str(device),
                "environment": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Running on {device}; results -> {results_dir}")

    rows: list[dict[str, float | int]] = []
    comparison_generator = torch.Generator(device=device).manual_seed(90_000 + config.seeds[0])
    comparison_noise = torch.randn(
        (2_500, 2), generator=comparison_generator, device=device
    )
    variation_examples: dict[float, Tensor] = {}
    for sigma_min in config.sigma_values:
        for seed in config.seeds:
            row, model, losses = run_single(
                config,
                seed=seed,
                sigma_min=sigma_min,
                device=device,
                results_dir=results_dir,
            )
            rows.append(row)
            if seed == config.seeds[0]:
                variation_examples[sigma_min] = sample_midpoint(
                    model,
                    comparison_noise,
                    n_function_evaluations=config.evaluation_nfe,
                ).cpu()
            if sigma_min == config.baseline_sigma and seed == config.seeds[0]:
                create_reproduction_figures(
                    model,
                    losses,
                    config,
                    seed=seed,
                    device=device,
                    results_dir=results_dir,
                )
    write_summary(rows, results_dir)
    plot_scatter_panels(
        [variation_examples[sigma] for sigma in config.sigma_values],
        [f"$\\sigma_{{min}}={sigma:.2f}$" for sigma in config.sigma_values],
        results_dir / "variation_samples.png",
    )
    print("Experiment complete. Inspect metrics.csv, summary.json, and the PNG figures.")


if __name__ == "__main__":
    main()
