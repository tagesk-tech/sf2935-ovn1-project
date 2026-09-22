# SF2935 ÖVN1 Project

Course project for **SF2935 — Modern Methods of Statistical Learning** at KTH Royal Institute of Technology, autumn 2026.

## Project

We study **Flow Matching for Generative Modeling** (Lipman et al., 2023). The code implements Conditional Flow Matching with the conditional optimal-transport probability path and trains a time-conditioned neural vector field on two-dimensional checkerboard data.

The reduced reproduction follows the mechanism of Figure 4 in the paper:

- visualize transport at times `0`, `1/3`, `2/3`, and `1`;
- compare midpoint-ODE samples at NFE `4`, `8`, `10`, and `20`; and
- report sliced Wasserstein distance against held-out checkerboard samples.

The project variation changes only `sigma_min`, the residual endpoint noise, using values `0.01`, `0.10`, and `0.30` over three fixed random seeds. This tests the tradeoff between a sharp endpoint and an easier, smoother transport.

## Repository structure

```text
code/data.py           checkerboard data handling
code/flow_matching.py  CFM loss, neural vector field, midpoint sampler
code/experiments.py    training, evaluation, plots, and experiment entry point
report/                 LaTeX report source and supplied NeurIPS 2026 template
results/                generated figures and metric tables
slides/                 presentation source
```

## Reproduction

Python 3.12 is recommended. From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python code/experiments.py
```

The last command runs the complete experiment: nine fits from three seeds crossed with three `sigma_min` values. It writes the configuration, raw metrics, summary, figures, and model checkpoints to `results/`.

No Flow Matching library is used. PyTorch supplies tensor operations, automatic differentiation, and Adam; the probability path, velocity target, training loss, and midpoint sampler are implemented directly.

## Output files

- `reproduction_trajectory.png`: evolution from Gaussian noise to data;
- `reproduction_nfe.png`: held-out target and four solver budgets;
- `nfe_metrics.csv`: quantitative comparison across NFE values;
- `variation_sigma_min.png`: mean and standard deviation across seeds;
- `variation_samples.png`: paired-noise comparison across `sigma_min`;
- `metrics.csv` and `summary.json`: raw and aggregated results;
- `baseline_training_loss.png`: optimization diagnostic.

## Authors

Group members will be added after the group is confirmed.

## Course integrity

This repository contains coursework. The submitted implementation, experiment design, analysis, and writing must be understood and defensible by both group members. Any permitted AI assistance will be disclosed in the report according to the course instructions.
