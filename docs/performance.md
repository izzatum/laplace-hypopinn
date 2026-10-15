# Validation and performance

The CPU reference environment is Python 3.12, PyTorch 2.5.1+cpu and NumPy
1.26.4. Numerical tests use small models for dense algebraic oracles; integration
and performance runs also exercise the notebook's 2,737-parameter Mish model.
The architecture remains configurable.

Run the full EX2 comparison (3,001 training updates, 1,000 historical diagonal
batches of 2,501 points, and 1,000 posterior samples):

```bash
python scripts/validate_notebook_reference.py --output /tmp/ex2-reference --threads 1
```

The script executes archived definition/data cells and the original diagonal
class, then runs the maintained package. It writes both scientific artifacts and
`comparison.json`: exact equality, maximum absolute error, RMSE, relative L2
error, explicit tolerances, stage timings, versions, source hashes and exit
status. Exit status 2 means at least one comparison exceeded its stated
tolerance. An existing complete `reference.pt` plus adjacent `reference.json`
can be reused with `--reference`; incompatible budgets are rejected. Reports
must be inspected before claiming reproduction, especially across backends.

The full CPU comparison on 2026-10-15 matched initial and trained parameters,
all 3,001 recorded losses, training/receiver data and the MAP field **bit for
bit**. The raw historical diagonal statistic differed by at most `7.45e-9`;
all 1,000 posterior coordinates differed by at most `1.20e-7` km (axis-coordinate
rounding). MAP was `(1.06, 1.52)` km and posterior mean approximately
`(1.05588, 1.54874)` km. Both ran with seed 4321 and one CPU thread.

| Stage | Archived reference | Maintained package |
| --- | ---: | ---: |
| Training | 33.30 s | 34.75 s |
| Historical diagonal fitting | 10.43 s | 11.67 s |
| Posterior prediction | 10.61 s | 11.80 s |

These are single full-run observations on an AMD EPYC 9V74 cloud CPU, not
warmed timing medians or a speedup claim. The maintained run also validates
finite values and accumulates online posterior field moments.

Measure individual estimators on the full notebook grids with short budgets:

```bash
python scripts/benchmark.py --output /tmp/benchmark-ex2 --threads 1
python scripts/benchmark.py --preset ex3 --velocity-path "$PWD/otway2.csv" \
  --output /tmp/benchmark-otway --threads 1
```

Each method runs in a separate process. Defaults retain the notebook model,
train for two updates on 32 collocation points, fit three historical batches or
one fixed weighted residual plan, and predict four posterior fields in chunks
of 4,096 grid points. INF uses rank four per layer; VaLLA uses four inducing
residuals and five variational updates. One warmup precedes three timed repeats;
the report preserves every duration and the median. CUDA timings synchronize
when explicitly requested. CPU peak RSS covers the entire worker, including
imports, training, warmup and repeats; it is not estimator-only memory.

Measured on the same CPU/environment on 2026-10-15, one thread. EX2 predicts
15,251 grid points; Otway predicts 75,030. Prediction time covers all four draws
and their online field moments. These warmed measurements cover maintained
implementations only; the archived comparison above was not warmed/repeated.

| Method | EX2 fit (ms) | EX2 predict (ms) | Otway fit (ms) | Otway predict (ms) | Peak RSS, EX2 / Otway (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical diagonal | 7.7 | 48.3 | 7.6 | 227.2 | 229 / 230 |
| GGN diagonal | 19.9 | 57.6 | 18.5 | 279.8 | 231 / 232 |
| GGN block diagonal | 77.8 | 57.3 | 79.0 | 271.3 | 260 / 260 |
| GGN full | 534.4 | 68.0 | 515.8 | 272.6 | 403 / 384 |
| KFAC | 16.4 | 56.6 | 15.2 | 265.7 | 230 / 231 |
| EFB | 33.5 | 63.1 | 27.1 | 258.2 | 230 / 231 |
| INF | 36.4 | 73.2 | 31.6 | 265.3 | 232 / 232 |
| VaLLA | 31.6 | 198.4 | 29.9 | 886.2 | 251 / 252 |

All sixteen method/grid combinations completed with finite predictions.

These short runs establish executable paths and measured throughput, not
accuracy or uncertainty calibration. Different approximations do different
amounts of work and define different covariances. Full Otway reproduction,
large-model scaling, GPU performance and repeated-run calibration studies have
not been established. Dense curvature uses an explicit allocation guard;
structured methods cache their smaller factors, and prediction streams fields
without retaining a sample-by-grid array unless disk-backed storage is requested.
