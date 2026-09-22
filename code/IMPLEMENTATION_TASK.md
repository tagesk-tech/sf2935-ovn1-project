# Flow Matching implementation

This project is a reduced reproduction of Figure 4 from Lipman et al., *Flow Matching for Generative Modeling*.

## Target

Train a neural vector field on two-dimensional checkerboard data using Conditional Flow Matching with a conditional optimal-transport path. Generate samples by integrating the learned ODE from Gaussian noise.

The method is implemented directly with PyTorch tensor operations and automatic differentiation; no Flow Matching library is used.

## Core training step

For a batch of checkerboard samples `x_data`:

```text
z       <- standard Gaussian noise with the same shape as x_data
tau     <- uniform times in [0, 1], one per example
x_tau   <- (1 - (1 - sigma_min) * tau) * z + tau * x_data
target  <- x_data - (1 - sigma_min) * z
loss    <- mean squared Euclidean distance between vector_field(tau, x_tau) and target
update the network parameters using the gradient of loss
```

These lines are the mathematical core. The rest of the code trains the field, integrates the ODE, computes the metric, and produces the report figures.

## Three-file implementation

1. `data.py`: checkerboard sampling and minibatches.
2. `flow_matching.py`: time-conditioned MLP, CFM loss, and midpoint ODE sampler.
3. `experiments.py`: training, sliced Wasserstein evaluation, plots, the endpoint-noise sweep, and the one-command entry point.

## Reproduction

- dataset: two-dimensional checkerboard;
- trajectory panels: `tau = 0, 1/3, 2/3, 1`;
- sampling panels: NFE `4, 8, 10, 20`;
- solver: fixed-step midpoint method;
- evidence: generated samples, sliced Wasserstein distance, training curve, and three fixed seeds.

## Project variation

Vary only `sigma_min` over `0.01`, `0.10`, and `0.30`. Smaller values target sharper endpoints; larger values smooth the endpoint and may make regression easier while blurring the checkerboard. Architecture, training budget, data, seeds, solver, and metric protocol remain fixed.

## Size

The complete implementation is 251 lines across the three Python files. It intentionally omits a general CLI, reusable package abstractions, and a separate validation framework so that the algorithm and experiment remain easy to read.
