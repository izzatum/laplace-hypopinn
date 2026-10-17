---
title: FAQ — PINN models, uncertainty, data and licensing
description: Answers about Laplace HypoPINN installation, PINN models, GGN and historical curvature, damping, CPU use, observed seismic data, and license scope.
---

# Frequently asked questions

## What does Laplace HypoPINN do?

Laplace HypoPINN learns a two-dimensional seismic travel-time field from a velocity
model and receiver travel times using a physics-informed neural network. It
estimates the hypocenter from the field minimum and applies local weight-space
uncertainty approximations to sample fields and source locations. The package is
intended for earthquake and microseismic localization research; the current
implementation uses `(z,x)` coordinates. Start with the [overview](index.md).

## How do I install Laplace HypoPINN?

Use Python 3.12 or later, clone
[`izzatum/laplace-hypopinn`](https://github.com/izzatum/laplace-hypopinn), create a
virtual environment, and run `python -m pip install -e '.[plot]'` from the
repository root. The [quick start](index.md#run-a-first-experiment) includes a
small CPU example. For interactive lessons, follow the additional Jupyter
instructions in the [tutorial guide](tutorials.md).

## Which Laplace approximations are implemented?

The eight method identifiers are `legacy_diagonal`, `diagonal`,
`block_diagonal`, `full`, `kfac`, `efb`, `inf` and `valla`. They cover the
historical loss-gradient statistic, direct generalized Gauss–Newton (GGN)
approximations, structured layer curvature and a fixed-inducing VaLLA adaptation.
Every method has an [executed tutorial](tutorials.md). The
[method reference](methods.md) defines their different approximation targets;
the methods are not interchangeable covariance estimators.

## How does historical diagonal curvature differ from GGN?

`legacy_diagonal` sums batch-size-weighted squares of gradients of the total
batch-mean loss, preserving the original notebook estimator. Because it squares
the summed gradient, it includes cross-residual and cross-family terms. It is
not the diagonal of the corrected GGN matrix, an exact Hessian or a
per-observation Fisher matrix. Corrected GGN uses weighted scalar-residual
Jacobians, `F = J.T @ diag(alpha) @ J`, and omits the residual-weighted second
derivatives of the exact loss Hessian. See the [equations](methods.md).

## Does `prior_precision` always define a Bayesian prior at a MAP solution?

No. In Laplace HypoPINN, posterior precision uses the convention
`P = scale * F + prior_precision * I`. The positive regularizer is post-hoc
damping unless training used the matching L2 penalty and likelihood scale.
The historical defaults and teaching notebooks use zero training prior, so their
final trained center need not be the maximum a posteriori (MAP) solution for the
regularized posterior being sampled. See [damping conventions](methods.md).

## Can I change the PINN depth, width or architecture?

Yes. The built-in multilayer perceptron supports configurable depth and widths,
with two inputs, one scalar output, and Mish, tanh or SiLU activations. A custom
model can provide a pure `forward(parameters, coordinates)` function or an
adapted `torch.nn.Module`. It must be deterministic, pointwise and differentiable
through the required coordinate and parameter derivatives. KFAC, EFB and INF
additionally require a compatible coordinate-derivative jet adapter. The
2,737-parameter notebook network is a preset and test fixture, not a required
architecture. See the [extension contracts](architecture.md).

## Is a GPU required?

No. The smoke example and all eight tutorials run on CPU. A supported CUDA device
can be selected explicitly for larger experiments, but GPU performance and
large-model scaling have not been established by the published repository
measurements. Dense full GGN has quadratic storage and cubic factorization cost;
its memory guard should be considered before increasing model size. Consult the
[performance guide](performance.md) for measured CPU results and limitations.

## Can I use observed receiver data without a known source?

Yes. Observed-data inversion requires a velocity grid, receiver coordinates and
receiver times; a true source and reference travel-time field are optional.
Use `(z,x)` in km, velocity in km/s and travel time in seconds. The
receiver values are travel times, not raw timestamp picks with an unresolved
event origin time. The
[observed-data example](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/extend_research.py)
shows the Python API. Analytic synthetic examples need no downloaded dataset.
The bundled Otway CSV has separate reuse terms described in the
[third-party notices](https://github.com/izzatum/laplace-hypopinn/blob/main/THIRD_PARTY_NOTICES.md).

## Do the tutorials reproduce the paper or provide calibrated intervals?

No. The tutorials use reduced budgets to demonstrate implementation and plots.
Their cross-section bands show sample mean plus or minus two sample standard
deviations, not validated 95% intervals. VaLLA uses coherent linearized
predictions; the other methods evaluate nonlinear models at sampled weights.
All report variation due to local network-weight uncertainty, which does not
by itself establish calibration for source location, velocity or arrival noise.
The [reproduction guide](reproducibility.md) defines historical settings and
the [performance guide](performance.md) identifies completed validation runs.

## What license and citation should I use?

The maintained package, documentation and new tutorials use the
[MIT License](https://github.com/izzatum/laplace-hypopinn/blob/main/LICENSE).
This grant excludes the historical archive and bundled velocity data; consult
the [third-party notices](https://github.com/izzatum/laplace-hypopinn/blob/main/THIRD_PARTY_NOTICES.md)
before reusing those materials. For research use, cite Izzatullah, Yildirim,
Waheed and Alkhalifah (2022), DOI
[10.1088/2632-2153/ac94b3](https://doi.org/10.1088/2632-2153/ac94b3), and the relevant
approximation sources. The [citation page](citation.md) includes BibTeX.
