# Laplace approximation tutorials

Each notebook is a standalone, executed lesson using the maintained package.
Saved figures are visible without running code. Start with the historical or GGN
diagonal notebook, then explore the structured methods.

| Notebook | What it demonstrates |
| --- | --- |
| [01 · Historical diagonal](01_legacy_diagonal.ipynb) | Compatibility statistic, resampled loss gradients and independent parameter draws |
| [02 · GGN diagonal](02_diagonal.ipynb) | Weighted residual Jacobians and diagonal precision |
| [03 · GGN block diagonal](03_block_diagonal.ipynb) | Exact within-layer GGN blocks and precision couplings |
| [04 · Full GGN](04_full.ipynb) | Between-layer couplings, Cholesky sampling and memory budgets |
| [05 · KFAC](05_kfac.ipynb) | Coordinate jets, Kronecker factors and precision spectra |
| [06 · EFB](06_efb.ipynb) | Fixed KFAC basis, residual replay and corrected eigenvalues |
| [07 · INF](07_inf.ipynb) | Selected eigenpairs, low-rank precision and diagonal-clipping diagnostics |
| [08 · VaLLA](08_valla.ipynb) | Fixed inducing residuals, variational fitting and coherent linearized predictions |

From the repository root, install and launch an environment:

```bash
python -m pip install -e '.[notebooks]' jupyterlab
python -m ipykernel install --sys-prefix --name laplace-hypopinn --display-name 'Laplace HypoPINN'
jupyter lab examples/notebooks
```

Select the **Laplace HypoPINN** kernel, then **Restart Kernel → Run All**.
An editor with Jupyter support can use the same environment. No GPU, downloaded
data, previous notebook, or cached training checkpoint is required. The examples
use an analytic gradient-velocity model; the original research notebooks remain
unchanged in [the archive](../../archive/README.md).

All eight notebooks were executed with Python 3.12 and CPU PyTorch 2.5.1; each
contains six rendered figures. CI also executes them to catch API regressions.

The tutorials retain the EX2 architecture (2,737 parameters), Mish activation,
seed 4321 and 101 × 151 grid. Teaching budgets are 800 updates, 512 collocation
points and 64 posterior draws. Historical diagonal uses 64 curvature batches;
INF retains up to eight directions per layer; VaLLA uses 32 inducing rows and
100 variational updates. Every notebook displays its configuration and versions.

Each lesson includes velocity/receiver geometry, convergence and receiver-fit
plots, reference/trained/error fields, method-specific diagnostics, field spread,
sampled source locations and a travel-time cross-section. Reference and trained
fields share a color scale, physical depth points downward, and checks verify
finite fields, positive precision and samples on the grid. The sampling API
reuses one parameter perturbation across every chunk of a complete field.

These reduced budgets demonstrate usage, not published reproduction or calibrated
uncertainty. Positive posterior regularization is post-hoc damping because the
training prior is zero. The cross-section band is sample mean ± two sample
standard deviations, not a 95% interval. VaLLA uses linearized predictions; the
other methods evaluate nonlinear models at sampled weights. Compare numerical
results only with those distinctions and the residual scaling in mind.

For new architectures, start from each notebook's configuration cell and rerun
training and fitting. Custom pointwise models follow the
[extension contract](../../docs/architecture.md); KFAC/EFB/INF require a compatible
jet adapter. See [method equations](../../docs/methods.md),
[reproduction settings](../../docs/reproducibility.md) and
[performance measurements](../../docs/performance.md) before increasing budgets.
