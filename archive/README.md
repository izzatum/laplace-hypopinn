# Historical notebooks and curvature implementation

The six original `EX_*.ipynb` notebooks are preserved byte-for-byte in
`notebooks/`. The original `curvature/` tree, including its bundled checkpoint,
is preserved in `legacy/curvature/`. `manifest.json` records each original path,
archived path and the SHA-256 recorded before relocation. The original
`otway2.csv` remains at the repository root. Source comments, attribution and
notebook outputs are retained. This historical material is excluded from the
maintained package's MIT grant. The archived curvature library retains its GPLv3
license and component notices; see [third-party notices](../THIRD_PARTY_NOTICES.md)
for license copies, provenance and data-reuse scope.

Use the installed package and its documented experiment presets for new work.
These files retain their historical scientific and runtime limitations: mutable
global state, expensive training loops, device assumptions, checkpoint writes,
legacy curvature names, and hardcoded predicted source coordinates in
`EX_3_plot.ipynb`. Its plotted coordinates are historical examples, not new
inference results. The historical diagonal statistic is the sum of batch size
times squared gradients of total mean loss; it is not an exact Fisher or Hessian.
The runner does not correct or rewrite archived code.

Install the project's notebook extra and register its kernel from the same
Python environment:

```bash
python -m pip install -e '.[notebooks]'
python -m ipykernel install --user --name laplace-hypopinn --display-name 'Laplace HypoPINN'
```

From the repository root, run a bounded import, seed, architecture-definition
and CSV smoke check (no training):

```bash
python scripts/run_archived_notebook.py EX_3.ipynb \
  --cells 2,4,6,10 --output /tmp/ex3-smoke-executed.ipynb
```

`--cells` uses original zero-based notebook cell indices, including markdown
cells, and executes them in the supplied order in one kernel. Dependencies must
be selected explicitly. Omit `--cells` to execute the entire notebook; this can
be expensive, can expose historical errors, and can write legacy checkpoints
in the repository root. Prefer selected smoke cells for routine validation.
The default kernel is `laplace-hypopinn`; select another with `--kernel NAME`.
The default cell timeout is 600 seconds; change it with `--timeout SECONDS`.

The kernel working directory is always the repository root so relative CSV
access works. Its `PYTHONPATH` starts with `archive/legacy` so original
`curvature` imports resolve, and bytecode generation is disabled to keep the
archive unchanged. Executed notebooks must be new files outside the repository;
the runner rejects existing outputs and all output paths inside the repository.
It clears prior outputs in the executed copy, retains original indices and
source text, and leaves archive inputs unchanged. Notebook code itself still
controls its own file writes; the runner is not a filesystem sandbox.

Check all preserved hashes and bounded execution with:

```bash
python -m pytest tests/test_archive.py -q
```
