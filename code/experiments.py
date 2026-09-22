"""Run the checkerboard reproduction and sigma_min variation."""

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import checkerboard, minibatch
from flow_matching import VectorField, cfm_loss, sample


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
SEEDS = (11, 22, 33)
SIGMAS = (0.01, 0.10, 0.30)
TRAIN_STEPS = 5_000
BATCH_SIZE = 512
N_TRAIN = 20_000
N_TEST = N_GENERATED = 5_000
NFE_VALUES = (4, 8, 10, 20)


def train(seed, sigma_min, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = checkerboard(N_TRAIN, 10_000 + seed, device)
    model = VectorField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator(device=device).manual_seed(20_000 + seed)
    losses = []

    for step in range(TRAIN_STEPS):
        x_data = minibatch(data, BATCH_SIZE, generator)
        loss = cfm_loss(model, x_data, sigma_min, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"seed={seed} sigma={sigma_min:.2f} step={step + 1} loss={loss.item():.4f}")
    return model.eval(), losses


@torch.inference_mode()
def sliced_wasserstein(x, y, seed, projections=256):
    n = min(len(x), len(y))
    generator = torch.Generator(device=x.device).manual_seed(seed)
    directions = torch.randn((projections, 2), generator=generator, device=x.device)
    directions /= directions.norm(dim=1, keepdim=True)
    x_proj = torch.sort(x[:n] @ directions.T, dim=0).values
    y_proj = torch.sort(y[:n] @ directions.T, dim=0).values
    return torch.sqrt(((x_proj - y_proj) ** 2).mean()).item()


def scatter_panels(samples, titles, filename):
    fig, axes = plt.subplots(1, len(samples), figsize=(3 * len(samples), 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, points, title in zip(axes, samples, titles):
        points = points.cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.45, rasterized=True)
        ax.set(title=title, xlim=(-5, 5), ylim=(-5, 5), xlabel="$x_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$x_2$")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_plot(losses):
    smooth = np.convolve(losses, np.ones(100) / 100, mode="valid")
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.plot(np.arange(99, len(losses)), smooth)
    ax.set(xlabel="Training step", ylabel="CFM loss", title="Training loss (100-step mean)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cpu")
    OUT.mkdir(exist_ok=True)
    rows, examples = [], {}
    comparison_noise = torch.randn(
        (2_500, 2), generator=torch.Generator(device=device).manual_seed(90_011), device=device
    )

    for sigma_min in SIGMAS:
        for seed in SEEDS:
            model, losses = train(seed, sigma_min, device)
            test = checkerboard(N_TEST, 30_000 + seed, device)
            generator = torch.Generator(device=device).manual_seed(40_000 + seed)
            noise = torch.randn((N_GENERATED, 2), generator=generator, device=device)
            generated = sample(model, noise, nfe=20)
            distance = sliced_wasserstein(test, generated, 50_000 + seed)
            rows.append((seed, sigma_min, distance, np.mean(losses[-100:])))
            torch.save(model.state_dict(), OUT / f"model_seed-{seed}_sigma-{sigma_min:.3f}.pt")

            if seed == 11:
                examples[sigma_min] = sample(model, comparison_noise, nfe=20).cpu()
            if seed == 11 and sigma_min == 0.10:
                figure_noise = torch.randn(
                    (2_500, 2), generator=torch.Generator(device=device).manual_seed(60_011), device=device
                )
                times = (0, 1 / 3, 2 / 3, 1)
                trajectory = [
                    figure_noise if t == 0 else sample(model, figure_noise, nfe=round(60 * t), end_time=t)
                    for t in times
                ]
                scatter_panels(trajectory, [f"$t={t:.2f}$" for t in times], "reproduction_trajectory.png")

                target = checkerboard(2_500, 70_011, device)
                nfe_samples = {nfe: sample(model, figure_noise, nfe=nfe) for nfe in NFE_VALUES}
                scatter_panels(
                    [target, *nfe_samples.values()],
                    ["Target data", *(f"NFE = {nfe}" for nfe in NFE_VALUES)],
                    "reproduction_nfe.png",
                )
                with (OUT / "nfe_metrics.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(("nfe", "sliced_wasserstein"))
                    for nfe, points in nfe_samples.items():
                        writer.writerow((nfe, sliced_wasserstein(target, points, 80_011)))
                training_plot(losses)

    scatter_panels(
        [examples[sigma] for sigma in SIGMAS],
        [f"$\\sigma_{{min}}={sigma:.2f}$" for sigma in SIGMAS],
        "variation_samples.png",
    )

    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("seed", "sigma_min", "sliced_wasserstein", "final_training_loss"))
        writer.writerows(rows)

    summary = {}
    for sigma in SIGMAS:
        values = [row[2] for row in rows if row[1] == sigma]
        summary[str(sigma)] = {
            "mean_sliced_wasserstein": float(np.mean(values)),
            "std_sliced_wasserstein": float(np.std(values, ddof=1)),
            "n_seeds": len(values),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    means = [summary[str(s)]["mean_sliced_wasserstein"] for s in SIGMAS]
    errors = [summary[str(s)]["std_sliced_wasserstein"] for s in SIGMAS]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.errorbar(SIGMAS, means, yerr=errors, marker="o", capsize=4)
    ax.set(xlabel="$\\sigma_{min}$", ylabel="Sliced Wasserstein distance", title="Endpoint smoothing variation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "variation_sigma_min.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    config = {
        "seeds": SEEDS,
        "sigma_values": SIGMAS,
        "baseline_sigma": 0.10,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_generated": N_GENERATED,
        "batch_size": BATCH_SIZE,
        "train_steps": TRAIN_STEPS,
        "learning_rate": 2e-3,
        "hidden_dim": 128,
        "n_hidden_layers": 3,
        "metric_projections": 256,
        "evaluation_nfe": 20,
        "panel_nfes": NFE_VALUES,
        "device": str(device),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Done. Results saved to {OUT}")


if __name__ == "__main__":
    main()
