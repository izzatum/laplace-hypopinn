"""Derivative-aware, weight-sharing KFAC with exact additive regularization.

This explicitly drops same-point cross-channel terms and separates activation
and residual-adjoint moments. It approximates the weighted residual GGN and is
not an exact Fisher or exact GGN estimator for a nonlinear PINN. Only factor
matrices/eigenvectors and product eigenvalues are stored, never dense Kronecker
matrices. Factors are eigensolved once, after all replayed residual chunks.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..model import flatten_parameters, predict
from .jets import residual_layer_terms
from .residuals import ResidualBatch


@dataclass(frozen=True)
class KFACFactor:
    A: Tensor
    G: Tensor
    exposure: Tensor
    QA: Tensor
    QG: Tensor
    a_eigenvalues: Tensor
    g_eigenvalues: Tensor
    eigenvalue_corrections: dict


def _psd_eigh(matrix):
    if not torch.isfinite(matrix).all():
        raise ValueError('nonfinite KFAC factor')
    symmetric = (matrix + matrix.T) * .5
    values, vectors = torch.linalg.eigh(symmetric)
    magnitude = values.abs().max() if values.numel() else matrix.new_zeros(())
    tolerance = torch.finfo(matrix.dtype).eps * max(matrix.shape[0], 1) * magnitude * 100
    if (values < -tolerance).any():
        raise ValueError('materially negative KFAC factor eigenvalue')
    correction = {'tolerance': float(tolerance), 'count': int((values < 0).sum()), 'magnitude': float(-values.clamp_max(0).sum())}
    return values.clamp_min(0), vectors, correction


def iter_layer_terms(parameters, residual_plan, chunk_size=64, *, forward=predict, jet_adapter=None):
    """Replay the fixed plan as bounded (batch, LayerTerms) pairs for EFB/INF."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if isinstance(residual_plan, ResidualBatch):
        residual_plan = (residual_plan,)
    for batch in residual_plan:
        for start in range(0, len(batch.coordinates), chunk_size):
            stop = start + chunk_size
            chunk = ResidualBatch(batch.family, batch.coordinates[start:stop], batch.targets.reshape(-1)[start:stop], batch.alpha[start:stop])
            yield chunk, residual_layer_terms(parameters, chunk, forward=forward, jet_adapter=jet_adapter)


def accumulate_factors(parameters, residual_plan, *, chunk_size=64, forward=predict, jet_adapter=None):
    """A=sum(alpha*a*a.T)/E, G=sum(alpha*b*b.T), E=sum(alpha*channels)."""
    _, layout = flatten_parameters(parameters)
    accumulators = None
    for batch, terms in iter_layer_terms(parameters, residual_plan, chunk_size, forward=forward, jet_adapter=jet_adapter):
        if len(terms.layers) != len(layout.blocks):
            raise ValueError('jet adapter layers must match canonical parameter blocks')
        if accumulators is None:
            accumulators = []
            for layer, block in zip(terms.layers, layout.blocks):
                if block.shape != (layer.adjoints.shape[-1], layer.activations.shape[-1]):
                    raise ValueError('jet adapter matrix shapes must match canonical parameter blocks')
                accumulators.append([layer.activations.new_zeros((layer.activations.shape[-1],) * 2), layer.adjoints.new_zeros((layer.adjoints.shape[-1],) * 2), batch.alpha.new_zeros(())])
        for layer, (a, g, exposure) in zip(terms.layers, accumulators):
            a.add_(torch.einsum('n,nsi,nsj->ij', batch.alpha, layer.activations, layer.activations))
            g.add_(torch.einsum('n,nsi,nsj->ij', batch.alpha, layer.adjoints, layer.adjoints))
            exposure.add_(batch.alpha.sum() * layer.activations.shape[1])
    if accumulators is None:
        raise ValueError('KFAC requires a nonempty residual plan')
    factors = []
    for a, g, exposure in accumulators:
        if not exposure > 0:
            raise ValueError('KFAC requires positive total residual exposure')
        a = a / exposure
        a = (a + a.T) * .5
        g = (g + g.T) * .5
        av, aq, ac = _psd_eigh(a)
        gv, gq, gc = _psd_eigh(g)
        factors.append(KFACFactor(a, g, exposure, aq, gq, av, gv, {'A': ac, 'G': gc}))
    return tuple(factors)


def fit_kfac(parameters, residual_plan, config, *, forward=predict, jet_adapter=None):
    """Fit factor eigenspaces and ``scale*g*a + prior_precision`` per layer."""
    from .base import Posterior

    factors = accumulate_factors(parameters, residual_plan, chunk_size=config.chunk_size, forward=forward, jet_adapter=jet_adapter)
    mean, layout = flatten_parameters(parameters)
    eigenbasis = tuple((factor.QG, factor.QA, config.scale * factor.g_eigenvalues[:, None] * factor.a_eigenvalues[None, :] + config.prior_precision) for factor in factors)
    return Posterior(
        mean=mean.detach().clone(), layout=layout, representation='kronecker',
        eigenbasis=eigenbasis, forward=forward,
        metadata={
            'method': 'kfac', 'target': 'weighted_residual_ggn_approximation',
            'approximation': 'first_order_jet_weight_sharing_moment_separation',
            'prior_precision': config.prior_precision, 'scale': config.scale,
            'regularization': 'additive_isotropic_precision; prior only if matching MAP training',
            'exposure': [float(f.exposure) for f in factors],
            'factor_eigenvalue_corrections': [f.eigenvalue_corrections for f in factors],
        },
    )
