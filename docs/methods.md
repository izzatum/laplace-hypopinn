# Curvature and predictive conventions

The [reproducibility contract](reproducibility.md) identifies the historical
settings and paper/code differences. This document defines the corrected research
target and its approximations. Method labels identify numerical conventions;
they do not assert calibrated uncertainty or reproduced published benchmarks.
Models may have configurable depth, width and activation, or custom differentiable
pointwise scalar forwards. Generic Jacobian methods do not depend on the notebook
network. Structured jet methods require a declared compatible model adapter.

## Residual target and precision

For parameters θ, coordinates q=(z,x), travel time Tθ(q), velocity v(q) and
observed receiver time t, use scalar residuals

\[
r_e=\|\nabla_q T_\theta\|^2-v^{-2},\qquad
r_d=T_\theta-t,\qquad r_+=2\operatorname{relu}(-T_\theta).
\]

Write the objective as

\[
\mathcal L(\theta)=\tfrac12\sum_i\alpha_i r_i(\theta)^2
 +\tfrac12\lambda_{\rm train}\|\theta\|^2,
\quad \alpha_i=2w_{\rm family}/n_{\rm family}.
\]

With unit family weights this matches the notebook means, including its positivity
penalty. The positivity residual is nonsmooth at zero; derivative checks avoid
that kink. Weights use **global family counts**, before chunking. Replaying a
weighted residual set must leave the estimator unchanged when chunk boundaries
change. A residual plan fixes selections or deterministic replay state; all passes
of a structured method use the same rows.

Let J have rows ∂rᵢ/∂θ. The corrected generalized Gauss–Newton (GGN) data target is

\[
F=J^\top\operatorname{diag}(\alpha)J,\qquad
P=sF+\lambda I,\quad s>0,\quad\lambda>0.
\]

Rows carry √αᵢ, **not residual values**. This PSD matrix excludes the
\(\sum_i\alpha_i r_i\nabla^2r_i\) term in the exact nonlinear-loss Hessian.
It can represent Gaussian residual-likelihood curvature under an explicitly
specified likelihood model; it is not automatically an empirical Fisher or a
calibrated physical likelihood.

The interface calls λ `prior_precision` and s `scale`. Describe λ as a Gaussian
prior only when training included the matching L2 objective and likelihood scale;
otherwise it is post-hoc damping. Training prior defaults to zero. Never add the
training prior twice. A Gaussian centered at the final trained iterate is a local
approximation; with post-hoc damping that iterate need not be the MAP of the
regularized posterior being sampled.

For structured affine layers, parameters use row-major vectorization of augmented
matrices `[W|b]`, concatenated by layer; consequently Kronecker ordering is G ⊗ A.
Custom parameter mappings use explicit recorded flatten/unflatten order.

## Historical diagonal and direct GGN methods

`legacy_diagonal` preserves the archived estimator

\[
f_{\rm legacy}=\sum_{b=1}^{B}n_b
 \big[\nabla_\theta\mathcal L_{\rm mean}^{(b)}\big]^{\odot2},
\qquad P_{\rm legacy}=\texttt{multiply}\,\operatorname{diag}(f_{\rm legacy})
 +\texttt{add}\,I.
\]

Each loss is the total of the notebook's mean PDE, receiver and positivity terms.
It includes cross-residual and cross-family effects from squaring the summed
loss gradient. It is not the diagonal of corrected F, a per-observation Fisher,
or an exact Hessian. Its dependence on the number/size of resampled batches is
part of compatibility. Draw independent layerwise Gaussian perturbations around
the same saved final parameters. Preserve the combined weight/bias noise layout
when comparing against [the archived implementation](../archive/legacy/curvature/curvatures.py).

| Identifier | Approximation to corrected F | Limitation |
| --- | --- | --- |
| `diagonal` | diag(F) | Discards all parameter correlations |
| `block_diagonal` | Exact blocks Fℓℓ per affine layer | Discards correlations between layers; custom mappings need an explicit partition |
| `full` | Complete dense F | Quadratic storage and cubic factorization; reference use with an explicit memory budget |

A full float64 2,737² matrix alone occupies about 57.2 MiB; factorization temporaries
require additional memory. The default per-allocation diagnostic guard is 256 MiB.
This is a budget, not a maximum supported model size. For dense precision
P=CCᵀ with lower-triangular Cholesky C, draw δ=C⁻ᵀξ. Solving Cδ=ξ would give the
wrong covariance in general.

## First-order coordinate jets and KFAC

Ordinary forward/backward hooks do not capture mixed coordinate/parameter
partials in the eikonal residual. Propagate value and coordinate-derivative
channels s∈{0,z,x} through each affine/Mish layer:

\[
z_0=Wa_0+b,\quad z_k=Wa_k,\quad
a'_0=\operatorname{Mish}(z_0),\quad
a'_k=\operatorname{Mish}'(z_0)\odot z_k.
\]

Here primes denote the next layer, not differentiation of an entire vector.
For a compatible alternative pointwise activation, replace Mish and its derivative
with that activation and its derivative; an unsupported adapter must fail explicitly.
Augment value-channel activations by 1 and derivative channels by 0 for biases.
For residual i let bᵢₛ=∂rᵢ/∂zₛ. Its exact layer Jacobian matrix is

\[
M_i=\sum_s b_{i,s}a_{i,s}^{\top}.
\]

Receiver and positivity rows use the value channel; PDE rows use all three.
Independent examples permit batched adjoints via an autograd call on the sum of
residuals. Mixed derivatives and this layer Jacobian must agree with a complete
autograd Jacobian reference.

`kfac` applies the explicit weight-sharing approximation

\[
E=\sum_i\alpha_i c_i,\quad
A=E^{-1}\sum_{i,s}\alpha_i a_{i,s}a_{i,s}^\top,\quad
G=\sum_{i,s}\alpha_i b_{i,s}b_{i,s}^\top,\quad
F_\ell^{\rm KFAC}=G\otimes A,
\]

where cᵢ is the number of active channels. This separates activation/adjoint
moments and drops same-point cross-channel terms; it is a jet-aware adaptation,
not exact GGN. It reduces to the ordinary single-channel formula for receiver-only
networks. Across layers, precision remains block diagonal.

With eigenvalues gⱼ,aₖ, add isotropic regularization exactly as
`s*g_j*a_k + prior_precision`. Independently damping the two factors produces a
different approximation. The conceptual source is Martens and Grosse (2015),
[Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://proceedings.mlr.press/v37/martens15.html).
That source does not establish this PINN jet adaptation or benchmark equivalence.

## EFB and sparse information form

`efb` freezes KFAC eigenvectors QG, QA, then replays the same residuals to compute

\[
\Lambda=\sum_i\alpha_i
  (Q_G^\top M_i Q_A)^{\odot2},\qquad Q=Q_G\otimes Q_A.
\]

Flattening Λ in canonical order equals diag(QᵀFℓℓQ). Thus EFB retains exact
GGN diagonal entries in this fixed basis, including the channel cross terms
omitted by KFAC, while discarding off-diagonal entries in that basis. Neither
reinitializing the accumulator per batch nor changing the basis during a pass is
valid. This is an EKFAC-style correction; see George et al. (2018),
[Fast Approximate Natural Gradient Descent in a Kronecker Factored Eigenbasis](https://arxiv.org/abs/1806.03884).
It is not a reimplementation of the faulty archived hook outputs.

`inf` retains at most the requested number of largest EFB eigenpairs **per layer**.
Build only the selected paired Kronecker vectors U, not the Cartesian closure of
selected factor vectors. With retained eigenvalues Λᵣ,

\[
d=\operatorname{diag}(F_{\ell\ell})-
  \operatorname{diag}(U\operatorname{diag}(\Lambda_r)U^\top),\qquad
F_\ell^{\rm INF}=\operatorname{diag}(\max(d,0))+
 U\operatorname{diag}(\Lambda_r)U^\top.
\]

The package's sparse information form follows the diagonal-plus-low-rank structure
in [archived INF](../archive/legacy/curvature/curvatures.py), with the exact
maintained definition in the [methods addendum](superpowers/specs/2026-10-15-laplace-methods-addendum.md).
Negative d need not be roundoff: EFB already discards off-diagonal information.
Clipping is an explicit PSD approximation. Report count and total magnitude of
clipped entries and requested/effective rank; retain raw statistics unchanged.
Add damping separately. INF does not generally recover dense F even at full rank,
because its EFB basis representation is diagonal.

For INF, and also VaLLA below, write precision P=D+BBᵀ with strictly positive
diagonal D. Whiten the columns C=D⁻¹ᐟ²B and cache a raw Householder QR
decomposition C=Q[R;0], followed by the small SVD R=U diag(s) Vᵀ. For a
standard normal parameter vector ξ, apply the inverse precision root:

\[
E=\operatorname{blockdiag}\!\left(
U\operatorname{diag}\!\left((1+s^2)^{-1/2}\right)U^\top,I\right),\qquad
\delta=D^{-1/2}QEQ^\top\xi.
\]

Then Cov(δ)=P⁻¹. Householder transforms apply Q and Qᵀ without materializing
a full parameter-space basis. The implementation scales informed components
directly using `1/hypot(1,s)`, preserving small variances without subtracting
nearly equal vectors or squaring large singular values. Factorization and
transforms use the declared dtype; no promotion or truncation is implicit.
Rank-deficient B and rank zero are valid. With k=min(parameter count, rank),
cache storage is O(parameter count × rank + k²); it is reused across draws.
Scale and regularization belong in D and B before sampling. Diagnostic dense
covariance roots apply this same transform only after the dense-memory guard.

## VaLLA: fixed-inducing PINN residual adaptation

`valla` adapts Luis A. Ortega, Simon Rodriguez-Santana and Daniel Hernandez-Lobato
(2024), [Variational Linearized Laplace Approximation for Bayesian Deep Learning](https://proceedings.mlr.press/v235/ortega24a.html),
ICML, PMLR 235. The [authors' implementation](https://github.com/Ludvins/Variational-LLA/blob/898ec63da3c1aa9c79f3c7d9d802453f83f80466/src/valla.py#L224)
is an additional equation source. Publication year is 2024; the preprint appeared
in 2023. This is the one newer extension in the maintained method inventory,
not an exhaustive survey of Laplace algorithms through 2026.

Fix the trained center, λ, likelihood weights and inducing functions. Whiten
residuals and their complete Jacobians by √(sαᵢ). Select distinct inducing
residual rows deterministically, recording family and source index. This differs
from ordinary network-output inducing variables in the publication.

For inducing Jacobians Jz and a learned lower-triangular factor L, define

\[
K_{zz}=J_zJ_z^\top/\lambda,\quad H=I+L^\top K_{zz}L,\quad
A=L\operatorname{solve}(H,L^\top).
\]

For query Jacobians Jx,

\[
K_{xx}=J_xJ_x^\top/\lambda,\quad K_{xz}=J_xJ_z^\top/\lambda,\quad
\operatorname{Cov}(f_x)=K_{xx}-K_{xz}AK_{xz}^\top.
\]

The mean remains the trained network output. The covariance-dependent KL is
½(logdet(H)−tr(Kzz A)); omitted mean terms are constant with the center/prior
fixed. For each whitened residual mean r and variance v, sum one of these terms
and add KL:

\[
\alpha_{\rm V}=0:\ \tfrac12(r^2+v),\qquad
\alpha_{\rm V}=1:\ \tfrac12\log(1+v)+\tfrac12r^2/(1+v).
\]

αV is VaLLA's divergence setting, distinct from residual weights αᵢ. Only 0 (ELBO
limit) and 1 are supported initially. Initialize L to identity: a zero factor can
have zero gradient. The research defaults are 32 inducing rows (capped by available
distinct residuals), 200 variational Adam steps, learning rate .01 and αV=0.
These are configurable budgets, not publication reproduction settings. Inducing
locations, prior precision and noise are fixed; no hyperparameter optimization
or learned inducing inputs is implied.

Equivalent parameter-perturbation covariance is

\[
C=(\lambda I+J_z^\top LL^\top J_z)^{-1}.
\]

Use the same low-rank precision sampler with D=λI and B=JzᵀL. Reuse **one** δ
across every chunk of a sampled grid and evaluate the linearized prediction
`T_MAP(q) + J_T(q) δ`, e.g. by JVP. Independent per-cell draws destroy the spatial
covariance required for localization; nonlinear evaluation at perturbed weights
is a different predictive convention. With every residual inducing, αV=0 and
L=I, the covariance equals `(λI + J.T@J)^-1`; optimization may change L.

## Numerical and interpretation limits

Symmetrize PSD estimates before eigendecomposition. Clamp only roundoff-scale
negative eigenvalues with recorded tolerance/correction; reject material negative
or nonfinite inputs. Use Cholesky solves and exact additive damping, with explicit
memory/chunk budgets. No silent device/dtype fallback or unrecorded jitter is part
of these definitions. Dense float64 algebraic identities, jet/autograd comparisons,
transformed-GGN EFB checks, small dense INF/VaLLA covariance and KL checks, and
chunk-coherent draws are the relevant numerical oracles; finite samples alone do
not establish correctness. A smoke run does not establish method quality.

All approximations are local to a trained center and chosen residual/likelihood
scales. Except for VaLLA's stated linearized convention, posterior parameter draws
are evaluated through the nonlinear model. Neither these methods nor the legacy
statistic by themselves establish calibrated uncertainty in seismic velocity,
arrival picks, or source location. Compare such claims against separate empirical
validation with recorded data and configuration.
