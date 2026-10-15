# Functional Laplace HypoPINN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an extensible functional research package with notebook-compatible experiments, corrected structured Laplace estimators, a validated recent research extension, archived notebooks and concise illustrated documentation.

**Architecture:** Pure tensor functions operate on explicit parameter trees and data. A small orchestration layer owns optimizer state, random generators, configuration and artifacts. Curvature structures share weighted residual Jacobians and a posterior sampling interface, while the historical batch-gradient diagonal remains a clearly named compatibility path.

**Tech Stack:** Python 3.12, PyTorch 2.5.1, NumPy 1.26.4, pytest, Ruff, setuptools/build; optional scikit-fmm and Matplotlib. CPU is required; CUDA is an optional validated backend.

**Spec:** `docs/superpowers/specs/2026-10-15-functional-research-design.md` and `docs/superpowers/specs/2026-10-15-laplace-methods-addendum.md`.

## Global Constraints

- Use the existing `/workspace/laplace-hypopinn` checkout; this cloud task is already isolated. Do not create a worktree unless explicitly requested.
- Preserve all six notebook files byte-for-byte in `archive/notebooks/`; record SHA-256 before moving them.
- Historical presets retain the six-hidden-layer, 2,737-parameter Mish network, seed 4321, 3,001 Adam updates and 2,501/5,001 collocation points.
- Coordinates are `(z,x)` in km; velocity is km/s and travel time is seconds.
- Historical diagonal: `sum(batch_size * grad(total_mean_loss)**2)`, precision `add + multiply * diagonal` with defaults in the approved spec.
- Corrected estimators use the positive semidefinite GGN of weighted scalar residuals; never silently label the legacy statistic an exact Fisher or Hessian.
- No mutable global model, RNG, device or configuration; no notebook execution or optional-dependency imports at package import.
- Keep float32 experiment defaults and float64 mathematical reference tests. No unmeasured speedup claims or unverified published-result claims.
- Select only the requested method and its prerequisites; keep file outputs out of the source tree and reject accidental overwrites.
- Paper inconsistencies are documented; the paper is a reference, not an obligation to reproduce erroneous signs or undocumented regularization.

## Review Focus

- Very unequal collocation/receiver counts: loss and GGN weighting must remain invariant to batch chunking and duplicating points with compensating weights (Tasks 3, 5).
- Near-singular or indefinite numerical factors: reject material invalidity, record any roundoff stabilization, and sample the inverse precision with the correct transpose (Tasks 5–8).
- Nonzero-origin, nonsquare custom grids and absent true source: preserve coordinate order and permit real-data inversion (Tasks 2, 10).
- Repeated runs, partial checkpoints and existing output paths: isolate RNG/optimizer state, reject incompatible resumes and avoid overwriting results (Tasks 1, 4, 10).
- Wheel installation away from the checkout and absent optional packages: work for gradient presets, accept explicit Otway data paths, and issue actionable dependency errors (Tasks 2, 11).

## Shared interfaces and file ownership

`types.py` defines `LayerParameters(weight: Tensor, bias: Tensor)` and `Parameters = tuple[LayerParameters, ...]`; layer weights have shape `(out,in)` and biases `(out,)`. Canonical vectorization is the row-major flattened augmented matrix `[W|b]` for each layer, concatenated in forward order. `ParameterLayout` records layer shapes/slices and round-trips this representation. `Streams` owns an explicitly supplied local NumPy `RandomState` and local PyTorch `Generator`; consuming these objects is intentional, while process-global RNG state is preserved.

`Grid(z: Tensor, x: Tensor, velocity: Tensor)` contains increasing axes and a `(nz,nx)` field. `Problem(grid, receiver_coordinates, receiver_times, reference_times=None, true_source=None)` contains observed or synthetic data. `TrainingBatch(coordinates, velocity, receiver_coordinates, receiver_times)` contains the fixed training subset. `LossTerms(total, pde, data, positivity, prior)` contains scalar tensors. Frozen configuration records expose every scientific choice and validate inputs before allocation.

`ResidualBatch` contains a residual family (`pde`, `data`, `positivity`), point inputs, targets and global per-residual weights `alpha`; `ResidualPlan` replays fixed weighted batches across multiple curvature passes. For objective `L=.5*sum(alpha_i*r_i**2)`, `jacobian_blocks(parameters, residual_batch)` yields weighted Jacobian rows in canonical parameter order, using chunks rather than a full data-by-parameter allocation. `Posterior` is a tagged data record for the selected representation; `draw_parameters(posterior, count, generator) -> ParametersBatch` returns explicit parameter tensors without mutating MAP weights. A `ParametersBatch` has the same layer structure with a leading sample dimension on each tensor. No covariance interface is imposed on methods that do not define one.

`Forward = Callable[[Parameters, Tensor], Tensor]`. `loss_terms`, `residual_values`, `jacobian_blocks`, `train`, and `run_experiment` accept keyword `forward: Forward = predict`; generic forward functions support diagonal/block/full and VaLLA via complete Jacobians. KFAC/EFB/INF initially require the documented Mish-MLP jet adapter and reject unsupported forwards explicitly. Numerical functions do not discover models through global registries.

## Task 1: Installable functional model and parameter/RNG contracts

**Files:** create `pyproject.toml`, `src/laplace_hypopinn/{__init__,types,config,model}.py`, `tests/{conftest,test_model,test_config}.py`, `tests/reference/notebook_adapter.py`. The adapter locates original or archived notebooks, executes only named definition cells and supplies the appropriate legacy module path; it does not run training at import.

**Interfaces:** produce `initialize_model(widths, initialization, streams, *, dtype, device) -> Parameters`, `predict(parameters, coordinates) -> Tensor`, `flatten_parameters(parameters) -> tuple[Tensor, ParameterLayout]`, and `unflatten_parameters(vector, layout) -> Parameters`. Configuration defaults come directly from the spec.

- [ ] Write failing tests: default parameter count is 2737; all four initializations agree with selected notebook definitions after matching default Linear draws; flatten/unflatten is exact; functional outputs/coordinate gradients match the notebook at fixed parameters; caller inputs/global RNG state remain unchanged; invalid widths and device selections fail before training.
- [ ] Run `python -m pytest tests/test_model.py tests/test_config.py -q`; confirm failures identify missing package behavior.
- [ ] Implement the typed parameter tree, pure linear/Mish forward, explicit initialization draws and immutable validated configurations. Match EX_1's sequential initialization order with shared local streams.
- [ ] Install editable with development extras; rerun the targeted tests until all pass.
- [ ] Commit this independently usable model API.

## Task 2: Velocity grids, synthetic references and observed-data inputs

**Files:** create `src/laplace_hypopinn/data.py`, `configs/{ex1,ex2,ex3,ex3_2,ex3_3}.toml`, `tests/test_data.py`.

**Interfaces:** consume model types/configuration; produce `prepare_problem(config) -> Problem`, `sample_training_batch(problem, count, streams) -> TrainingBatch`, and `load_observed_problem(path) -> Problem`. Custom NPZ input keys are `z`, `x`, `velocity`, `receiver_coordinates`, `receiver_times`; true-source/reference arrays are optional.

- [ ] Write failing tests for analytic constant/linear-gradient travel times; exact gradient grid/receiver indices; Otway CSV checksum, transpose, endpoints and float32 floor-source index; no source requirement for observed input; rejection of nonpositive velocity, unordered axes, mismatched arrays, out-of-domain receivers and oversampling without replacement.
- [ ] Run `python -m pytest tests/test_data.py -q` and confirm the intended failures.
- [ ] Implement data preparation and validated presets, lazily importing scikit-fmm only for Otway. Resolve explicit data paths against the configuration location, without relying on the current working directory. Freeze verified Otway bounds, not a fragile floating-point `arange` construction.
- [ ] Run targeted tests with and without optional fast-marching support; verify errors explain the required extra.
- [ ] Commit data preparation and presets.

## Task 3: Losses, residual weights and derivative-aware Jacobians

**Files:** create `src/laplace_hypopinn/losses.py`, `src/laplace_hypopinn/curvature/{__init__,residuals,jacobians}.py`, `tests/{test_losses,test_jacobians}.py`.

**Interfaces:** consume `predict`, `TrainingBatch`, parameter layout; produce `loss_terms(parameters, batch, config) -> LossTerms`, `make_residual_plan(problem, config, streams) -> ResidualPlan`, `residual_values(parameters, batch) -> Tensor`, and `jacobian_blocks(parameters, batch, chunk_size) -> Iterator[Tensor]`.

- [ ] Write failing tests comparing all loss components/parameter gradients to notebook cell 8; prove `L=.5*sum(alpha*r*r)` with `alpha=2*term_weight/global_term_count`; verify the positivity residual `2*relu(-T)` and explicit zero/nonzero L2 prior; compare mixed parameter/coordinate derivatives to float64 finite differences away from the positivity kink.
- [ ] Add tests that unequal receiver/PDE counts, chunk sizes and duplicate points with halved weights yield identical GGN; global prior is added exactly once and is not counted as data curvature.
- [ ] Run `python -m pytest tests/test_losses.py tests/test_jacobians.py -q` and observe the intended failures.
- [ ] Implement pure residual functions and chunked `torch.func` Jacobians. Separate the historical whole-batch-gradient path from per-residual GGN; preserve fixed residual selection/weights across multi-pass estimators.
- [ ] Rerun targeted tests, including an observed-only problem with no reference map.
- [ ] Commit loss and Jacobian foundations.

## Task 4: Training and reproducible checkpoints

**Files:** create `src/laplace_hypopinn/{training,artifacts}.py`, `tests/{test_training,test_checkpoint}.py`.

**Interfaces:** consume model, batch and loss functions; produce `train(parameters, batch, config, streams, *, resume=None) -> TrainingResult`, `save_checkpoint(path, state) -> None`, and `load_checkpoint(path, expected_config) -> TrainingState`. State includes parameters, optimizer/scheduler state, step count, local RNG states and schema version.

- [ ] Write failing tests for one-step Adam/cosine parity with notebook code; unchanged caller initial parameters; two-run RNG independence; uninterrupted versus resumed runs; incompatible layout/config rejection; nonfinite-loss errors identifying the step; safe output overwrite handling.
- [ ] Run `python -m pytest tests/test_training.py tests/test_checkpoint.py -q` and confirm failures.
- [ ] Implement locally owned optimizer tensors with Adam `.005`, betas `(.5,.9)`, 3001-step cosine scheduling; detach coordinate leaves between iterations to avoid unused gradient accumulation; collect detached metrics and transfer at logging intervals. Include explicit prior regularization only when configured.
- [ ] Rerun targeted tests and a short EX_2 fit; confirm checkpoint round-trip and finite loss history.
- [ ] Commit training and artifact schema.

## Task 5: Historical diagonal and corrected diagonal/block/full references

**Files:** create `src/laplace_hypopinn/curvature/{base,diagonal,dense,sampling}.py`, `tests/curvature/{test_diagonal,test_dense,test_sampling}.py`.

**Interfaces:** produce `fit_legacy_diagonal(parameters, problem, config, streams) -> Posterior`, `fit_ggn(parameters, residual_plan, *, structure, prior_precision, scale, max_dense_bytes) -> Posterior`, and the shared `draw_parameters` dispatcher. Dense diagnostic functions return precision only below an explicit byte limit.

- [ ] Write failing tests for legacy diagonal accumulation and layerwise weight/bias noise ordering; corrected diagonal equals `diag(J.T@J)`; block structure equals the layer blocks of the dense matrix; covariance root satisfies `S@S.T = precision^-1`; unequal term counts and chunking preserve scale; rank-deficient data with positive prior remains valid.
- [ ] Add rejection tests for negative/nonfinite scales, nonpositive precision and requests exceeding the dense-memory limit. Check roundoff jitter is recorded and never silently used to hide substantial indefiniteness.
- [ ] Run `python -m pytest tests/curvature/test_diagonal.py tests/curvature/test_dense.py tests/curvature/test_sampling.py -q` and confirm failures.
- [ ] Implement streamed accumulators, exact additive damping and Cholesky solves. Sample from precision `L@L.T` using `solve(L.T, noise)`; never substitute the transposed legacy covariance-root multiplication. Default dense-memory cap is 256 MiB with an explicit override.
- [ ] Rerun targeted tests; verify finite nonmutating samples and empirical covariance with seeded confidence tolerances in addition to deterministic algebraic tests.
- [ ] Commit the basic estimators and posterior sampling contract.

## Task 6: Correct derivative-aware KFAC

**Files:** create `src/laplace_hypopinn/curvature/{jets,kfac}.py`, `tests/curvature/{test_jets,test_kfac}.py`.

**Interfaces:** produce `jet_forward(parameters, coordinates) -> JetTrace`, `residual_layer_terms(parameters, residual_batch) -> LayerTerms`, and `fit_kfac(parameters, residual_plan, config) -> Posterior`. `LayerTerms` contains value/z/x channel activations and residual adjoints with the bias channel convention defined in the addendum.

- [ ] Write failing tests that summed channel outer products reconstruct every layer's exact residual Jacobian for nonlinear eikonal, receiver and positivity terms; include different receiver and collocation batch sizes to catch the old overwritten-hook bug.
- [ ] Add tests for single affine receiver-model exactness, explicitly formed Kronecker factor agreement, data-duplication invariance, PSD factors, exact additive-prior covariance and canonical vectorization.
- [ ] Run `python -m pytest tests/curvature/test_jets.py tests/curvature/test_kfac.py -q` and confirm failures.
- [ ] Implement first-order jet propagation and the documented jet/weight-sharing KFAC approximation with explicit exposure weights; do not use mutable forward/backward hooks. Eigensolve each factor once and apply damping to the product eigenvalues.
- [ ] Rerun tests against dense tiny matrices; check that ordinary receiver-only networks reduce to the standard single-channel formula.
- [ ] Commit corrected KFAC with documented approximation semantics.

## Task 7: EFB/EKFAC eigenbasis correction

**Files:** create `src/laplace_hypopinn/curvature/efb.py`, `tests/curvature/test_efb.py`.

**Interfaces:** consume fixed KFAC eigenspaces and replayable exact Jacobian blocks; produce `fit_efb(parameters, residual_plan, kfac_basis, config) -> Posterior`.

- [ ] Write failing tests that EFB eigenvalues equal the diagonal of the dense per-layer GGN in the fixed Kronecker eigenbasis; ensure two residual batches both contribute and the basis is not recomputed/reset between batches; ensure weighting/chunking are invariant.
- [ ] Run `python -m pytest tests/curvature/test_efb.py -q` and confirm failures.
- [ ] Implement a second accumulation pass using exact mixed-derivative layer Jacobians; cache eigenspaces, use consistent row-major weight/bias layout and apply one explicit prior/scale convention.
- [ ] Verify covariance roots and seeded samples against dense reconstructed EFB precision.
- [ ] Commit EFB with its relationship to EKFAC explicitly documented.

## Task 8: Stable low-rank INF

**Files:** create `src/laplace_hypopinn/curvature/inf.py`, `tests/curvature/test_inf.py`.

**Interfaces:** consume corrected diagonal/EFB statistics; produce `fit_inf(diagonal, efb, config) -> Posterior` with selected eigenpairs, diagonal remainder, clipping diagnostics and small cached factorization.

- [ ] Write failing tests for an exact rank budget, canonical Kronecker eigenpair selection, zero-rank diagonal limit, repeated/zero eigenvalues, negative diagonal-remainder diagnostics and material invalid-input rejection. Require no Cartesian-product rank expansion.
- [ ] Test precision solves and sampling-factor covariance against explicit `D + U@diag(lambda)@U.T` on small float64 cases; specifically exercise singular low-rank Gram matrices with valid positive total precision.
- [ ] Run `python -m pytest tests/curvature/test_inf.py -q` and confirm failures.
- [ ] Implement the documented nonnegative diagonal correction and exact additive prior; cache Woodbury solves and use perturb-and-solve sampling rather than the legacy chain of matrix inverses. Record clipped count/mass and effective rank in artifacts.
- [ ] Rerun targeted tests and a short nonlinear PINN integration for all corrected structures.
- [ ] Commit INF.

## Task 9: VaLLA fixed-inducing PINN adaptation

**Files:** create `src/laplace_hypopinn/curvature/valla.py`, `tests/curvature/test_valla.py`, `configs/research/valla.toml`.

**Interfaces:** consume weighted residual plans, Jacobians and low-rank sampling; produce `fit_valla(parameters, residual_plan, config, streams) -> Posterior`, `valla_objective(factor, cached_batch, prior_precision, alpha) -> Tensor`, and `linearized_predict(parameters, perturbation, coordinates) -> Tensor`.

- [ ] Write failing tests for the author covariance/KL equations against explicit dense weight covariance; `alpha=0` and `alpha=1` data terms; all-residuals-inducing and `L=I` exact covariance oracle; PSD and unchanged trained mean; deterministic inducing selection and frozen MAP/prior/inducing functions.
- [ ] Add tests for complete residual weighting with mixed value/derivative inputs, finite variational gradients, reducing the deterministic objective on a tiny fixture, and rejecting unsupported alpha values/nonpositive prior.
- [ ] Run `python -m pytest tests/curvature/test_valla.py -q` and confirm failures.
- [ ] Implement the equations in the addendum using Cholesky solves; optimize only a lower-triangular factor initialized to identity. Use 32 inducing rows, 200 Adam steps at .01 and `alpha=0` as configurable research defaults. Cache inducing Jacobians plus whitened residual means, prior variance diagonals and residual-to-inducing kernel rows; discard full residual Jacobians after building those caches. Never form grid-by-grid covariance.
- [ ] Verify coherent parameter perturbations and JVP predictions give identical locations/maps across different grid chunk sizes. Record objective history, inducing identities and the linearized prediction convention.
- [ ] Commit VaLLA with original citation and explicit PINN-adaptation limits.

## Task 10: Integrated experiments, prediction and artifact workflows

**Files:** create `src/laplace_hypopinn/{experiment,prediction,cli,__main__}.py`, `tests/{test_experiment,test_prediction,test_cli}.py`, `examples/extend_research.py`.

**Interfaces:** produce `run_experiment(config, *, problem=None) -> ExperimentResult`, `predict_grid(parameters, grid, chunk_size) -> Tensor`, `sample_locations(posterior, grid, config, generator) -> PosteriorSummary`, and CLI `hypopinn run --config PATH --output DIR`, `hypopinn presets`, `hypopinn plot RUN_DIR`. Posterior summary includes locations, mean, population standard deviation and optional online field moments.

- [ ] Write failing integration tests for all five research presets with short budgets; all eight estimator identifiers; EX_1 sequential initialization; custom observed-data inversion without true-source fields; first-grid-minimum ties; nonzero grid origins; unchanged MAP parameters and global RNG.
- [ ] Test coherent VaLLA versus nonlinear sampled-network prediction routing, grid/posterior chunk invariance, output collisions, artifact schemas and resumable versus incompatible checkpoints. Never silently switch methods after a failure.
- [ ] Run `python -m pytest tests/test_experiment.py tests/test_prediction.py tests/test_cli.py -q` and confirm failures.
- [ ] Compose the existing stage functions. Save configuration, source/version/data provenance, method-specific precision/normalization diagnostics, timing, checkpoint, loss history, MAP field and posterior location samples. Stream field moments and make complete field storage an explicit disk-backed option.
- [ ] Verify CLI and Python extension example; test user-supplied velocity/data and forward-model callables with clear limits for jet-specific KFAC support. Unsupported model operations must fail explicitly rather than give incorrect curvature.
- [ ] Commit the executable research workflow.

## Task 11: Byte-preserving archive, plots and concise documentation

**Files:** move `EX_*.ipynb` to `archive/notebooks/`, move `curvature/` to `archive/legacy/curvature/`; create `archive/{README.md,manifest.json}`, `scripts/run_archived_notebook.py`, `src/laplace_hypopinn/plotting.py`, `docs/{architecture,reproducibility,methods,performance}.md`, `tests/{test_archive,test_plotting,test_install}.py`; replace `README.md`; add citation metadata without inventing software-license permissions.

**Interfaces:** `plot_run(run_directory, output_directory) -> list[Path]`; archival runner uses the repository root as kernel cwd and adds `archive/legacy` to the kernel module path. The runner writes executed copies outside the archive and supports selected cells for smoke tests.

- [ ] Record SHA-256 of every notebook and legacy source before moving. Write failing tests that the manifest matches after relocation, all six notebooks are present and the input bytes stay unchanged. Test archived imports and CSV access using the documented runner.
- [ ] Write failing tests for headless plots from real artifact data; figures must not use EX_3_plot's hardcoded predicted coordinates. Add wheel installation checks outside the checkout, with explicit data-path and missing-optional-extra cases.
- [ ] Run `python -m pytest tests/test_archive.py tests/test_plotting.py tests/test_install.py -q` and confirm failures.
- [ ] Move files without rewriting notebooks or legacy source. Implement plots/archival runner, concise README with Mermaid workflow and quick/full commands, and detailed scientific-difference/method documentation. Preserve all source attribution and document legacy limitations.
- [ ] Run archive smoke, installed CLI, extension example and README commands; inspect generated plots and confirm axis/units/source markers.
- [ ] Commit archive and documentation.

## Task 12: Full validation, performance evidence, CI and environment refresh

**Files:** create `scripts/{validate_reproduction,benchmark_curvature}.py`, `.github/workflows/tests.yml`; extend `tests/reference/notebook_adapter.py` and update `docs/performance.md`, `.gitignore` and existing environment configuration through the supported draft tool.

**Interfaces:** validation scripts accept output directory, seed, device, thread count and explicit budgets; write machine-readable results with source versions and underlying command exit status. Reference adapters execute selected archived definitions and the original diagonal statistic, omitting unrelated KFAC/EFB/INF work when comparing that statistic.

- [ ] Add tests for report schemas, missing/incomplete-run rejection and correct numeric comparison; configure CI to run fast tests/lint/package build on Python 3.12 CPU. Mark long research runs opt-in.
- [ ] Run `python -m pytest`, `ruff check .`, `python -m build` and isolated wheel CLI checks. Fix failures before expanding runs; do not disable scientific assertions to pass.
- [ ] Execute EX_2 with 3001 training updates, 1000 curvature batches and 1000 posterior draws in the same pinned CPU environment for both reference and new compatibility path; record forward/loss/parameter/location and posterior comparisons, separately identifying any floating-point/backend variance. Report train, curvature and sampling outcomes individually.
- [ ] Execute short complete gradient/Otway runs for every maintained estimator and inspect finite artifacts, uncertainty diagnostics and method conventions. Record full Otway/all-method experiments as unrun unless actually executed.
- [ ] Benchmark warmed-up legacy/new diagonal, corrected structured fitting and streaming prediction on fixed data, with controlled threads and synchronized CUDA if available. Record median timings and peak memory, numerical agreement and the limits of the comparison. Profile before requesting RunPod; no paid provisioning without a concrete authorized budget.
- [ ] Refresh reusable environment instructions for the installed package and new archive paths. Exercise installation, startup/readiness commands and save `install_script`/`start_skill` through the environment draft tool; saving does not publish the environment.
- [ ] Obtain whole-change code/scientific review, address findings, rerun affected checks, and commit validated completion. Report tests, measured performance, methods implemented, limitations, configuration fields saved and any unrun long experiments.

## Execution recommendation

Use **subagent-driven execution** with narrowly assigned file ownership and a fresh
reviewer for each numerical component. The residual/Jacobian/layout contracts are
shared sequential prerequisites; only independent work after those contracts are
stable should run concurrently. Method covariance mistakes can produce plausible
plots, so independent mathematical review is worth the extra review cost.

The user approved this plan and requires flexible models. Implementation uses
the recommended subagent-driven approach. Named parameter mappings and custom
forward adapters supersede the earlier fixed layer tuple in all interfaces;
notebook architecture is retained only as an example preset and test fixture.
