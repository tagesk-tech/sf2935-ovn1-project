# Flow Matching implementation task

This is the collaborative specification for a reduced reproduction of Figure 4 from Lipman et al., *Flow Matching for Generative Modeling*.

## Target

Train a neural vector field on two-dimensional checkerboard data using Conditional Flow Matching with the conditional optimal-transport path. Generate samples by integrating the learned ODE from Gaussian noise.

Do not copy an existing Flow Matching implementation. General-purpose tensor operations, automatic differentiation, optimizers, and plotting libraries are allowed by the assignment.

## Core training step

For a batch of checkerboard samples `x_data`:

```text
z       <- standard Gaussian noise with the same shape as x_data
tau     <- uniform times in [0, 1], one per example
x_tau   <- (1 - (1 - sigma_min) * tau) * z + tau * x_data
target  <- x_data - (1 - sigma_min) * z
pred    <- vector_field(tau, x_tau)
loss    <- mean squared Euclidean distance between pred and target
update the network parameters using the gradient of loss
```

The six lines above are the mathematical core. The model, optimizer, sampler, tests, configuration, metrics, plots, and command-line entry point are supporting experimental machinery.

## Components to implement

1. `data.py`: deterministic two-dimensional checkerboard sampler.
2. `model.py`: small multilayer perceptron mapping `(tau, x_tau)` to a two-dimensional velocity.
3. `train.py`: the Conditional Flow Matching training step above.
4. `sample.py`: fixed-step midpoint integration of `dx/dtau = vector_field(tau, x)`.
5. `metrics.py`: sliced Wasserstein distance between held-out data and generated samples.
6. `plots.py`: sample snapshots at several times and panels for several function-evaluation budgets.
7. `run_all.py`: one command that trains, evaluates, and regenerates every result.
8. `tests/`: analytic training-pair check, deterministic seed check, shape checks, and a zero-vector-field sampler check.

## Reproduction

Reproduce the mechanism of Figure 4 rather than its full ImageNet-scale results:

- dataset: two-dimensional checkerboard;
- method: Flow Matching with conditional OT paths;
- trajectory panels: `tau = 0, 1/3, 2/3, 1`;
- sampling panels: several small function-evaluation budgets, initially `4, 8, 10, 20`;
- solver: fixed-step midpoint method;
- evidence: generated sample plots, sliced Wasserstein distance, training curves, and at least three random seeds.

## Proposed project variation

Vary only `sigma_min`. Predict the tradeoff before running:

- smaller `sigma_min` should target a sharper endpoint distribution;
- it may also demand a more accurate learned field or numerical solution;
- larger `sigma_min` should smooth the endpoint and may make fitting easier while blurring the checkerboard.

Hold architecture, training budget, dataset, seeds, solver, and metric protocol fixed.

## Verification gates

- Hand-compute one `(x_tau, target)` pair and match the program exactly.
- On a single Gaussian target, verify that the learned transport moves mass in the expected direction.
- Confirm that generated samples change continuously as the ODE step count increases.
- Report all selected seeds, not only the best-looking run.
- Record every departure from the paper in the report.

## Estimated size

- mathematical training core: roughly 10 lines;
- neural vector-field model: roughly 20–35 lines;
- midpoint sampler: roughly 15–25 lines;
- complete reproducible experiment: roughly 180–300 clear lines, excluding tests and comments.

Short code is plausible. Short reasoning is not.

