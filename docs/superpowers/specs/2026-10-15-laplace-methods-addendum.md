# Maintained Laplace methods and numerical corrections

This addendum incorporates the user's approved expansion: implement KFAC, EFB,
INF and up to two newer algorithms; fix discovered defects; use the paper as a
reference; assess local compute before requesting RunPod. It supplements the
functional package design, whose architecture and scientific compatibility
requirements remain in effect.

## Method inventory and scope

| Identifier | Curvature / prediction convention |
| --- | --- |
| `legacy_diagonal` | Original squared batch-loss gradients and layerwise Gaussian draws; named compatibility mode |
| `diagonal` | Diagonal of weighted residual generalized Gauss–Newton (GGN) |
| `block_diagonal` | Exact per-layer GGN blocks |
| `full` | Dense GGN reference, guarded by a memory limit |
| `kfac` | Explicit first-order coordinate-jet, weight-sharing KFAC approximation |
| `efb` | Fixed Kronecker eigenbasis with exact residual-Jacobian diagonal correction (EKFAC-style) |
| `inf` | Rank-limited EFB eigenspace plus nonnegative diagonal correction |
| `valla` | Variational Linearized Laplace Approximation, ICML 2024; fixed-inducing PINN adaptation |

Implement one newer algorithm, VaLLA. “Up to two” is a maximum, not a reason to
add a second overlapping or weakly justified method. VaLLA adds a genuinely distinct
function-space approximation with a small variational factor. It is not a claim
to implement every Laplace method published through 2026.

The existing classifier-specific convolution/attention utilities are outside the
PINN package's scope. Preserve the original `curvature/` directory under
`archive/legacy/curvature/`, with no new runtime imports from it. This prevents
unfixed historical code from masquerading as the maintained implementation.

## A shared corrected curvature target

For scalar residuals write

`L(theta) = .5 * sum_i alpha_i * r_i(theta)^2 + .5 * lambda_train * ||theta||^2`.

For the notebook's mean-square terms, `alpha_i=2*w_term/n_term`. Residuals are
`||grad_x T||^2 - velocity^-2`, `T_receiver - observed`, and `2*relu(-T)`.
The last residual preserves the notebook positivity loss. Its nonsmooth boundary
is documented; finite-difference derivative tests avoid the kink.

The corrected data curvature is `F = J.T @ diag(alpha) @ J`, not the Hessian of
the full nonlinear loss and not the squared gradient of its batch mean. Do not
multiply Jacobian rows by the residual value. Final precision is
`P = scale * F + precision_regularizer * I`, with positive regularizer, positive
scale and explicit metadata. In implementation interfaces `prior_precision`
names this regularizer; describe it as a Gaussian prior only when training used
the matching L2 term and likelihood scale. Otherwise label it post-hoc damping.
The historical presets keep zero training prior and their original posterior
regularization; they do not acquire an undocumented MAP objective.

Global residual counts determine weights before chunking. Replaying the same
weighted residual set must give the same estimator regardless of chunk boundaries.
The residual plan stores selections or deterministic replay state so that KFAC,
EFB and INF passes use identical data. New research algorithms do not inherit the
legacy quadratic scaling from multiplying sums of batch-mean factors.

Use row-major vectorization of each augmented matrix `[W|b]`, concatenated by
layer. This gives Kronecker ordering `G ⊗ A`. A separately tested compatibility
adapter maps any historical serialization layouts to this canonical form.

## KFAC for a derivative loss

Ordinary neural-network hooks are insufficient: the eikonal loss differentiates
outputs with respect to coordinates and then differentiates those derivatives
with respect to weights. The old single-record hooks also overwrite collocation
activations with receiver activations. The maintained path has no such hooks.

Propagate first-order jets for value and each coordinate direction:

`z0 = W a0 + b`, `zk = W ak`, `a0_next = Mish(z0)`,
`ak_next = Mish'(z0) * zk` for `k in {z,x}`.

Append 1 to the value-channel activation and 0 to derivative-channel activations
for bias augmentation. For residual i, let `b_i,s = d r_i / d z_s`. Its exact
layer Jacobian matrix is `M_i = sum_s b_i,s @ a_i,s.T`. Receiver/positivity
residuals use the value channel; PDE residuals use all three channels. Because
examples are independent, batched residual adjoints can be obtained from one
autograd call on their sum. Validate this construction against `torch.func`
Jacobian references, including all mixed derivatives.

Use the explicitly documented weight-sharing approximation:

`E = sum_i alpha_i * active_channels_i`

`A = sum_i,s alpha_i * outer(a_i,s, a_i,s) / E`

`G = sum_i,s alpha_i * outer(b_i,s, b_i,s)`

`F_kfac = G ⊗ A`.

This drops same-point cross-channel terms and separates activation and adjoint
moments. It is a jet-aware adaptation of KFAC, not an assertion of exact GGN or
of reproducing a published PINN-KFAC benchmark. It reduces to the ordinary
single-channel formula for receiver-only networks. Compute exact isotropic
regularization in the Kronecker eigenbasis: eigenprecisions are
`scale*g_j*a_k + prior_precision`. Adding square-root damping to each factor
is a different approximation and is not the default.

## EFB and INF

Freeze KFAC eigenvectors `QG, QA`. In a second pass over the same residuals,
accumulate `Lambda = sum_i alpha_i * (QG.T @ M_i @ QA)^2`. The acceptance oracle
is the diagonal of `Q.T @ F_block @ Q`, where `Q=QG ⊗ QA`. This includes the
jet-channel cross terms omitted by the KFAC approximation. Never reinitialize
EFB inside the batch loop or change its eigenbasis during accumulation.

INF selects at most the configured number of largest EFB eigenpairs per layer.
Construct only those paired Kronecker vectors; do not take a Cartesian closure
that exceeds the rank budget. Let selected vectors/eigenvalues be `U, Lambda_r`:

`d = diag(F_block) - diag(U @ diag(Lambda_r) @ U.T)`

`F_inf = diag(max(d, 0)) + U @ diag(Lambda_r) @ U.T`.

Negative corrections can be substantive because EFB discards off-diagonal
information. Clipping is an explicit PSD approximation: report count, total
clipped magnitude and requested/effective rank. Do not mutate stored raw
statistics. Add prior/damping separately.

For precision `P=D+B@B.T`, with positive diagonal D, cache a Cholesky of
`I+B.T@D^-1@B`. Draw independent standard normals xi and eta and use

`h = sqrt(D)*xi + B@eta`

`delta = D^-1@h - D^-1@B @ solve(I+B.T@D^-1@B, B.T@D^-1@h)`.

This perturb-and-solve draw has covariance `P^-1`. It handles rank-deficient B
without requiring a Cholesky of `B.T@B`, uses no explicit inverses and avoids a
dense parameter covariance.

## VaLLA: a bounded recent extension

Reference: Luis A. Ortega, Simon Rodriguez-Santana and Daniel Hernandez-Lobato,
[Variational Linearized Laplace Approximation for Bayesian Deep Learning](https://proceedings.mlr.press/v235/ortega24a.html),
ICML 2024 (preprint 2023). Equations were verified against the
[authors' implementation](https://github.com/Ludvins/Variational-LLA/blob/898ec63da3c1aa9c79f3c7d9d802453f83f80466/src/valla.py#L224)
and [author publication metadata](https://github.com/Ludvins/Ludvins.github.io/blob/master/_data/publications.yml).
The [Laplace library documentation](https://github.com/aleximmer/Laplace/blob/main/docs/valla.md)
independently identifies the method. Publisher/arXiv access was unavailable here;
accessible author-maintained sources establish this choice, not an exhaustive
literature survey.

Fix trained parameters, prior precision, residual weights and inducing functions.
Use whitened residuals `r_tilde_i=sqrt(alpha_i)*r_i` and their complete parameter
Jacobians; a global curvature/likelihood scale is folded into these weights.
Select inducing residual rows deterministically, preserving their family and
source index in artifacts. This extends ordinary-output VaLLA to local linearized
PINN residuals and must be labeled as such.

With inducing Jacobians Jz, positive prior precision lambda, and learned lower
triangular L:

`Kzz = Jz @ Jz.T / lambda`

`H = I + L.T @ Kzz @ L`

`A = L @ solve(H, L.T)`.

For query Jacobians Jx, `Kxx=Jx@Jx.T/lambda`, `Kxz=Jx@Jz.T/lambda`, and
`covariance = Kxx - Kxz @ A @ Kxz.T`. The predictive mean remains the trained
network output. The covariance-dependent KL is
`.5*(logdet(H) - trace(Kzz@A))`; omitted mean terms are constant because the
center and prior are fixed.

For whitened residual mean r and variance v, optimize summed terms plus KL:

- `alpha=0`: `.5*(r*r+v)` (ELBO limit).
- `alpha=1`: `.5*log1p(v)+.5*r*r/(1+v)`.

Initially support only these two alpha values. Initialize the triangular factor
to identity, not zero (which can give zero parameter gradients). Keep inducing
locations, prior and noise fixed. Do not claim hyperparameter optimization,
learned inducing inputs, or the original publication's benchmark reproduction.
Default research budget: 32 inducing rows, 200 variational Adam steps at .01,
`alpha=0`, recorded and configurable; cap count at available distinct residuals.

Equivalent weight-perturbation covariance is
`C = (lambda*I + Jz.T @ L @ L.T @ Jz)^-1`.
Use the same stable low-rank precision sampler as INF. Each draw supplies one
parameter perturbation reused across all grid chunks. Predict linearly as
`T_MAP(x) + J_T(x) @ delta` (JVPs avoid materializing the full grid Jacobian).
Independently sampling each grid cell would destroy the spatial dependence
needed for hypocenter argmin and is forbidden. Nonlinear weight-sampled
predictions are not interchangeable with this VaLLA predictive convention.

## Numerical safeguards and validation

- Symmetrize PSD estimates before eigendecomposition; clamp only roundoff-scale
  negative eigenvalues and report the tolerance/correction. Reject material
  negative/nonfinite input; no silent CPU fallback or dtype/device changes.
- Use Cholesky solves, exact additive regularization and bounded chunks; no
  `.data`, unrecorded jitter, repeated matrix inverses or mutated MAP models.
- Correct the legacy block sampler orientation: for `P=L@L.T`, draw
  `solve(L.T, normal)`, whose covariance is `P^-1`.
- Test dense algebraic covariance identities in float64, not merely whether
  a sample is finite. Add seeded empirical checks as supporting evidence.
- Validate jet Jacobians against autograd, EFB against a transformed dense GGN,
  and INF/VaLLA against explicit small precision matrices.
- VaLLA with all residuals inducing, `alpha=0`, and `L=I` must have covariance
  `(lambda*I+J.T@J)^-1`. Also verify KL against the dense Gaussian formula,
  coherent samples under grid rechunking, and both value and derivative rows.
- All methods need a short complete PINN run; one full historical EX_2 run retains
  the earlier reproduction commitment. Long comparisons across methods and
  Otway are separately timed and reported, not implied by smoke tests.

## Compute and delivery

The network has 2,737 parameters: a full float64 P×P matrix is approximately
57.2 MiB before temporary factorizations. CPU can validate all methods with tiny
references and representative experiments. Keep a 256 MiB per-allocation guard
for dense diagnostics and use chunked Jacobians in production estimators.

Do not provision paid RunPod resources or request secret values now. After local
profiling, request a concrete GPU type/time budget only if larger research runs
need it. CPU correctness is mandatory; GPU throughput is a separate validation
claim. All new code, CI, archival migration, README and environment startup
instructions must be verified together before declaring the refactor complete.
