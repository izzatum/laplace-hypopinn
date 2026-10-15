"""Fixed-inducing residual VaLLA with coherent linearized PINN predictions.

Ortega, Rodriguez-Santana and Hernandez-Lobato, ICML 2024:
https://proceedings.mlr.press/v235/ortega24a.html
Author equations: https://github.com/Ludvins/Variational-LLA/blob/
898ec63da3c1aa9c79f3c7d9d802453f83f80466/src/valla.py

This adaptation uses weighted value/derivative residual functionals. MAP, prior,
weights and inducing rows stay fixed; only a triangular covariance factor is
optimized. It does not optimize inducing inputs or hyperparameters, or reproduce
publication benchmarks. A prior unmatched to training is post-hoc damping.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from ..model import flatten_parameters, predict, unflatten_parameters
from .base import Posterior, guard_dense, require_scalar
from .jacobians import iter_jacobians
from .residuals import ResidualBatch, residual_values


@dataclass(frozen=True)
class VaLLACache:
    """Detached fixed-MAP statistics; no full residual-by-parameter Jacobian.

    Identities are (plan batch index, family, source row index). Kernel rows and
    residual means include the full sqrt(scale*alpha_i) whitening. Frozen records
    protect structure; callers must not mutate the contained tensor snapshots.
    """

    residual_mean: Tensor
    prior_variance: Tensor
    kernel_xz: Tensor
    kernel_zz: Tensor
    inducing_jacobian: Tensor
    inducing_identities: tuple[tuple[int, str, int], ...]
    prior_precision: float
    inducing_features: Tensor
    feature_projection: Tensor
    perpendicular_variance: Tensor


def _positive_count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def build_valla_cache(parameters, residual_plan, config, streams, *, forward=predict):
    """Select fixed distinct rows with the local NumPy stream and stream caches.

    Only inducing rows (M by P), one bounded Jacobian chunk, residual means,
    prior variance diagonals and N by M cross-kernel rows are retained. Additional
    N by min(M,P) feature projections and perpendicular norms permit positive
    variance evaluation without subtracting nearly equal large kernels. QR and
    variance algebra explicitly use float64 on the original device, including
    for float32 models. Public results and posterior tensors retain model dtype.
    This function snapshots tensors and never differentiates the fixed MAP.
    """
    require_scalar(config.prior_precision, 'prior_precision', positive=True)
    require_scalar(config.scale, 'scale', positive=True)
    _positive_count(config.inducing_count, 'inducing_count')
    _positive_count(config.chunk_size, 'chunk_size')
    if not hasattr(streams, 'numpy'):
        raise ValueError('an explicit local stream with a NumPy generator is required')
    local = {name: value.detach().clone() for name, value in parameters.items()}
    mean, layout = flatten_parameters(local)
    if not torch.isfinite(mean).all():
        raise ValueError('MAP parameters must be finite')
    if isinstance(residual_plan, ResidualBatch):
        residual_plan = (residual_plan,)
    plan = tuple(ResidualBatch(b.family, b.coordinates.detach().clone(),
                               b.targets.detach().clone(), b.alpha.detach().clone())
                 for b in residual_plan)
    total = sum(b.alpha.numel() for b in plan)
    if total < 1:
        raise ValueError('VaLLA requires at least one residual row')
    count = min(config.inducing_count, total)
    feature_count = min(count, layout.numel)
    for numel in (count*layout.numel, total*count, count*count,
                  min(total, config.chunk_size)*layout.numel, total,
                  (count+feature_count)*feature_count):
        guard_dense(numel, torch.float64, config.max_dense_bytes)
    selected = sorted(int(i) for i in streams.numpy.choice(total, count, replace=False))
    inducing, identities = [], []
    offset = 0
    scale_root = math.sqrt(float(config.scale))
    for batch_index, batch in enumerate(plan):
        source_rows = [index-offset for index in selected
                       if offset <= index < offset+batch.alpha.numel()]
        if source_rows:
            indices = torch.tensor(source_rows, dtype=torch.long, device=batch.coordinates.device)
            subset = ResidualBatch(batch.family, batch.coordinates[indices],
                                   batch.targets.reshape(-1)[indices], batch.alpha[indices])
            inducing.extend(rows.detach()*scale_root for rows in iter_jacobians(
                local, subset, config.chunk_size, forward=forward))
            identities.extend((batch_index, batch.family, row) for row in source_rows)
        offset += batch.alpha.numel()
    inducing = torch.cat(inducing)
    prior = float(config.prior_precision)
    kernel_zz = inducing@inducing.T/prior
    # Store compact Householder reflectors, never the P by P orthogonal matrix.
    # QR is fixed; only the later small factor-space algebra is differentiated.
    packed, tau = torch.geqrf(inducing.T.to(torch.float64)/math.sqrt(prior))
    reflectors = packed[:, :feature_count].contiguous()
    inducing_features = packed[:feature_count].triu().contiguous()
    # Preallocate only the permitted O(N M) and O(N) persistent caches.
    cross = mean.new_empty((total, count))
    variance = mean.new_empty(total)
    residual_mean = mean.new_empty(total)
    projection = torch.empty((total, feature_count), dtype=torch.float64, device=mean.device)
    perpendicular = torch.empty(total, dtype=torch.float64, device=mean.device)
    offset = 0
    for batch in plan:
        for chunk_index, rows in enumerate(iter_jacobians(
                local, batch, config.chunk_size, forward=forward)):
            rows = rows.detach()*scale_root
            start = chunk_index*config.chunk_size
            stop = start+rows.shape[0]
            subset = ResidualBatch(batch.family, batch.coordinates[start:stop],
                                   batch.targets.reshape(-1)[start:stop], batch.alpha[start:stop])
            residual = residual_values(local, subset, forward=forward).detach()
            target = slice(offset+start, offset+stop)
            cross[target] = rows@inducing.T/prior
            variance[target] = rows.square().sum(1)/prior
            residual_mean[target] = residual*subset.alpha.sqrt()*scale_root
            transformed = torch.ormqr(reflectors, tau,
                                      (rows.to(torch.float64)/math.sqrt(prior)).T.contiguous(),
                                      left=True, transpose=True)
            projection[target] = transformed[:feature_count].T
            perpendicular[target] = transformed[feature_count:].square().sum(0)
        offset += batch.alpha.numel()
    if not all(torch.isfinite(value).all() for value in
               (inducing, kernel_zz, cross, variance, residual_mean,
                inducing_features, projection, perpendicular)):
        raise ValueError('VaLLA residual means or Jacobians contain nonfinite values')
    return VaLLACache(residual_mean, variance, cross, kernel_zz, inducing,
                      tuple(identities), prior, inducing_features, projection, perpendicular)


def valla_terms(factor, cached_batch, prior_precision, *, return_diagnostics=False):
    """Return function variance diagonals and the covariance-dependent Gaussian KL.

    If Jz.T/sqrt(prior)=Q R, the feature precision is S=I+R L L.T R.T.
    Variance is ||q_perp||^2 + q_core.T solve(S,q_core), a sum of positive
    norms. An augmented QR provides its triangular precision square root without
    forming a possibly ill-conditioned Gram matrix. This is algebraically the
    author's H/A covariance and KL, with no large-kernel subtraction. Fixed
    float64 work preserves gradients for strongly informed float32 models; the
    returned tensors retain factor dtype. Upper factor entries are ignored.
    """
    require_scalar(prior_precision, 'prior_precision', positive=True)
    if float(prior_precision) != cached_batch.prior_precision:
        raise ValueError('prior_precision is fixed when building the cache')
    size = cached_batch.kernel_zz.shape[0]
    if factor.shape != (size, size) or not torch.isfinite(factor).all():
        raise ValueError('variational factor must be a finite inducing-by-inducing matrix')
    if factor.dtype != cached_batch.kernel_zz.dtype or factor.device != cached_batch.kernel_zz.device:
        raise ValueError('variational factor and cache dtype/device must match')
    lower = factor.tril().to(torch.float64)
    core = cached_batch.inducing_features@lower
    feature_count = core.shape[0]
    identity = torch.eye(feature_count, dtype=lower.dtype, device=lower.device)
    # The identity rows preserve prior directions even when core is rank deficient
    # and its norm is too large for I + core@core.T to represent the prior.
    _, precision_root = torch.linalg.qr(torch.cat((core.T, identity)), mode='reduced')
    solved = torch.linalg.solve_triangular(precision_root.T,
                                           cached_batch.feature_projection.T, upper=False)
    variance = cached_batch.perpendicular_variance+solved.square().sum(0)
    tolerance = 50*size*torch.finfo(variance.dtype).eps*cached_batch.prior_variance.max()
    if (variance < -tolerance).any() or not torch.isfinite(variance).all():
        raise ValueError('VaLLA covariance has material negative or nonfinite variance')
    diagnostics = {
        'variance_clipped_count': int((variance < 0).sum().detach()),
        'variance_clipped_magnitude': float((-variance.clamp_max(0)).sum().detach()),
        'variance_roundoff_tolerance': float(tolerance.detach()),
    } if return_diagnostics else None
    inverse_root = torch.linalg.solve_triangular(precision_root, identity, upper=True)
    kl = .5*(2*precision_root.diag().abs().log().sum()-feature_count
             +inverse_root.square().sum())
    variance, kl = variance.to(factor.dtype), kl.to(factor.dtype)
    return (variance, kl, diagnostics) if return_diagnostics else (variance, kl)


def valla_objective(factor, cached_batch, prior_precision, alpha):
    """Summed alpha=0 ELBO-limit or alpha=1 data terms plus the fixed-center KL."""
    if alpha not in (0, 1):
        raise ValueError('alpha must be 0 or 1')
    variance, kl = valla_terms(factor, cached_batch, prior_precision)
    squared_mean = cached_batch.residual_mean.square()
    if alpha == 0:
        data = .5*(squared_mean+variance)
    else:
        data = .5*torch.log1p(variance)+.5*squared_mean/(1+variance)
    return data.sum()+kl


def fit_valla(parameters, residual_plan, config, streams, *, forward=predict):
    """Optimize only L, returning the low-rank precision lambda I+Jz.T L L.T Jz."""
    if config.alpha not in (0, 1):
        raise ValueError('alpha must be 0 or 1')
    _positive_count(config.variational_steps, 'variational_steps')
    require_scalar(config.variational_lr, 'variational_lr', positive=True)
    # Snapshot mean before cache construction so caller parameters never receive gradients.
    mean, layout = flatten_parameters(parameters)
    mean = mean.detach().clone()
    cache = build_valla_cache(parameters, residual_plan, config, streams, forward=forward)
    count = cache.inducing_jacobian.shape[0]
    factor = torch.eye(count, dtype=mean.dtype, device=mean.device).requires_grad_()
    optimizer = torch.optim.Adam([factor], lr=config.variational_lr)
    history = []
    for step in range(config.variational_steps+1):
        objective = valla_objective(factor, cache, config.prior_precision, config.alpha)
        if not torch.isfinite(objective):
            raise ValueError('nonfinite VaLLA variational objective')
        history.append(float(objective.detach()))
        if step == config.variational_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if not torch.isfinite(factor.grad).all():
            raise ValueError('nonfinite VaLLA variational gradient')
        optimizer.step()
    lower = factor.detach().tril().clone()
    _, _, diagnostics = valla_terms(lower, cache, config.prior_precision,
                                    return_diagnostics=True)
    metadata = {
        **diagnostics,
        'estimator': 'fixed_inducing_weighted_residual_valla',
        'prior_precision': float(config.prior_precision), 'scale': float(config.scale),
        'alpha': config.alpha, 'inducing_identities': cache.inducing_identities,
        'requested_inducing_count': config.inducing_count, 'effective_inducing_count': count,
        'variational_steps': config.variational_steps, 'variational_lr': config.variational_lr,
        'variational_factor': lower, 'objective_history': tuple(history),
        'prediction_convention': 'T_MAP(x) + J_T(x) @ delta; one delta per complete field',
        'fixed': ('MAP', 'prior_precision', 'residual_weights', 'inducing_rows'),
        'citation': 'Ortega et al., Variational Linearized Laplace Approximation, ICML 2024',
        'citation_url': 'https://proceedings.mlr.press/v235/ortega24a.html',
        'adaptation_limits': 'Fixed inducing PINN residual functionals; no inducing/hyperparameter optimization or publication benchmark claim',
        'variance_roundoff_policy': 'Positive QR feature-space variance; no subtractive cancellation or clipping',
        'variance_work_dtype': 'float64 (same device); public results retain model dtype',
        'jitter': 0.,
    }
    return Posterior(mean, layout, 'low_rank', diagonal=torch.full_like(mean, config.prior_precision),
                     low_rank=(cache.inducing_jacobian.T@lower).detach(), metadata=metadata,
                     forward=forward, prediction_kind='linearized')


def linearized_predict(parameters, perturbation, coordinates, *, forward=predict):
    """MAP plus a parameter JVP, reusing the same flat delta over all grid chunks."""
    mean, layout = flatten_parameters(parameters)
    if perturbation.shape != mean.shape or not torch.isfinite(perturbation).all():
        raise ValueError('perturbation must be one finite canonical parameter vector')
    if perturbation.dtype != mean.dtype or perturbation.device != mean.device:
        raise ValueError('perturbation and parameters dtype/device must match')
    def evaluate(vector):
        output = forward(unflatten_parameters(vector, layout), coordinates)
        if output.shape not in ((coordinates.shape[0],), (coordinates.shape[0], 1)):
            raise ValueError('forward must return one scalar per point')
        return output
    center, tangent = torch.func.jvp(evaluate, (mean,), (perturbation,))
    return center+tangent
