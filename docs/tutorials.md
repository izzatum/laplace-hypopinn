---
title: Eight Laplace approximation tutorials with PyTorch PINNs
description: Executed Python notebooks for historical diagonal, GGN diagonal, block, full, KFAC, EFB, INF and VaLLA, with seismic travel-time plots and diagnostics.
---

# Laplace approximation tutorials

These eight standalone Jupyter notebooks demonstrate Laplace HypoPINN from data
preparation through training, curvature fitting, posterior sampling and plotting.
Read the HTML lessons below to see the explanations, code and six saved figures
without running Python. Each lesson also links its editable notebook on GitHub. Start with GGN diagonal to learn the corrected residual-Jacobian
workflow, or historical diagonal to study notebook compatibility.

| HTML lesson and notebook | What you will implement and inspect |
| --- | --- |
| [01 · Historical diagonal](https://izzatum.github.io/laplace-hypopinn/tutorials/legacy_diagonal/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/01_legacy_diagonal.ipynb) | Fit the archived loss-gradient statistic from resampled batches; compare marginal parameter spread with damping alone. |
| [02 · GGN diagonal](https://izzatum.github.io/laplace-hypopinn/tutorials/diagonal/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/02_diagonal.ipynb) | Build weighted residual Jacobians and diagonal precision; inspect precision and marginal standard deviations. |
| [03 · GGN block diagonal](https://izzatum.github.io/laplace-hypopinn/tutorials/block_diagonal/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/03_block_diagonal.ipynb) | Retain exact within-layer GGN blocks; visualize precision couplings while omitting between-layer correlations. |
| [04 · Full GGN](https://izzatum.github.io/laplace-hypopinn/tutorials/full/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/04_full.ipynb) | Fit dense GGN with a memory budget; inspect between-layer couplings and sample using Cholesky factors. |
| [05 · KFAC](https://izzatum.github.io/laplace-hypopinn/tutorials/kfac/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/05_kfac.ipynb) | Use coordinate jets to fit Kronecker factors for derivative residuals; inspect factor and product spectra. |
| [06 · EFB](https://izzatum.github.io/laplace-hypopinn/tutorials/efb/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/06_efb.ipynb) | Replay residuals in a fixed KFAC eigenbasis; inspect the corrected GGN diagonal entries in that basis. |
| [07 · INF](https://izzatum.github.io/laplace-hypopinn/tutorials/inf/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/07_inf.ipynb) | Select EFB eigenpairs and fit diagonal-plus-low-rank precision; examine retained rank and clipped diagonal remainder. |
| [08 · VaLLA](https://izzatum.github.io/laplace-hypopinn/tutorials/valla/) · [Notebook](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/08_valla.ipynb) | Fit fixed inducing residuals variationally; plot objective history and coherent linearized predictive fields. |

The [method reference](methods.md) gives the equations and source papers. EFB is
an EKFAC-style correction; the maintained KFAC, INF and VaLLA implementations
have explicit PINN-specific conventions. They are research adaptations, not
claims to reproduce every result of their source publications.

## Run a notebook

After the [repository installation](index.md#run-a-first-experiment), activate
the same virtual environment and run these commands from the repository root:

```bash
python -m pip install -e '.[notebooks]' jupyterlab
python -m ipykernel install --sys-prefix --name laplace-hypopinn --display-name 'Laplace HypoPINN'
jupyter lab examples/notebooks
```

Choose the **Laplace HypoPINN** kernel and run **Restart Kernel → Run All**.
No GPU, downloaded data, earlier notebook or cached checkpoint is required. The
analytic velocity model supplies synthetic receiver observations and reference
fields. The [notebook index in the repository](https://github.com/izzatum/laplace-hypopinn/blob/main/examples/notebooks/README.md)
records the tested execution environment.

## Read the figures and assumptions

Each lesson plots velocity and receiver geometry, convergence and receiver fit,
reference and trained fields with signed error, method-specific diagnostics,
field spread and sampled source locations, and a travel-time cross-section.
Coordinates are in km and travel time in seconds; depth increases downward.

The teaching configuration retains the EX2 Mish network with 2,737 parameters,
seed 4321 and a 101 × 151 grid. It uses 800 training updates, 512 collocation
points and 64 posterior draws. Historical diagonal uses 64 curvature batches,
INF retains up to eight directions per layer, and VaLLA uses 32 inducing rows
with 100 variational updates. These are teaching budgets, not full reproduction
settings.

Posterior regularization is post-hoc damping because training uses no parameter
prior. The cross-section band is sample mean ± two sample standard deviations,
not a calibrated 95% interval. VaLLA evaluates linearized predictions; all other
lessons evaluate nonlinear models at sampled weights. Each complete sampled
field reuses one parameter perturbation across its grid chunks.

## Extend a lesson

Change the configuration cell and rerun training and fitting to investigate
different widths, depth or budgets. The [architecture guide](architecture.md)
defines custom-model contracts, including the coordinate-derivative adapter
needed for KFAC/EFB/INF. Consult [performance measurements](performance.md)
before scaling dense methods and [reproduction settings](reproducibility.md)
before comparing with the historical experiments. The
[original notebooks](https://github.com/izzatum/laplace-hypopinn/blob/main/archive/README.md)
remain preserved separately.
