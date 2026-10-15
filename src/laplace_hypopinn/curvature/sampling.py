"""Exact precision-based Gaussian draws with explicit local random generators."""

import torch

from ..model import unflatten_parameters
from .base import DEFAULT_MAX_DENSE_BYTES, cholesky_with_jitter, guard_dense


def _generator(streams, generator):
    if generator is not None and streams is not None:
        raise ValueError('supply streams or generator, not both')
    generator = getattr(streams, 'torch', streams) if generator is None else generator
    if generator is None:
        raise ValueError('an explicit local random generator is required')
    return generator


def sample_vectors(posterior, count, streams=None, *, generator=None):
    """Return (count, parameters) vectors; never alter the mean or caller parameters."""
    if not isinstance(count, int) or count < 1:
        raise ValueError('sample count must be a positive integer')
    generator = _generator(streams, generator)
    mean = posterior.mean
    def normal(shape):
        return torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=generator)
    kind = posterior.representation
    if kind in {'diagonal', 'block_diagonal', 'kronecker'}:
        # Sample-major, then layer-major calls preserve the historical W|b noise order.
        samples = []
        for _ in range(count):
            pieces = []
            for index, block in enumerate(posterior.layout.blocks):
                noise = normal(block.shape)
                if kind == 'diagonal':
                    piece = noise.reshape(-1)/posterior.diagonal[block.slice].sqrt()
                elif kind == 'block_diagonal':
                    piece = torch.linalg.solve_triangular(posterior.blocks[index].T,
                                                          noise.reshape(-1, 1), upper=True).reshape(-1)
                else:
                    qg, qa, eigenvalues = posterior.eigenbasis[index]
                    piece = (qg@(noise/eigenvalues.sqrt())@qa.T).reshape(-1)
                pieces.append(piece)
            samples.append(torch.cat(pieces))
        offsets = torch.stack(samples)
    elif kind == 'dense':
        offsets = torch.stack([
            torch.linalg.solve_triangular(posterior.factor.T,
                                          normal((mean.numel(), 1)), upper=True).reshape(-1)
            for _ in range(count)
        ])
    elif kind == 'low_rank':
        offsets = torch.stack([
            _low_rank_inverse_root(posterior, normal((mean.numel(), 1))).reshape(-1)
            for _ in range(count)
        ])
    else:
        raise ValueError(f'unsupported posterior representation {kind!r}')
    return (offsets+mean).detach()


def _low_rank_inverse_root(posterior, noise):
    """Apply D^-1/2 Q diag(U diag(1/hypot(1,s)) U.T, I) Q.T.

    Raw Householder reflectors apply Q without a dense parameter-space basis.
    Rotating and scaling directly avoids subtracting nearly equal vectors.
    """
    if posterior.low_rank.shape[1]:
        qr, tau = posterior._low_rank_qr, posterior._low_rank_tau
        rotation, scales = posterior._low_rank_rotation, posterior._low_rank_scales
        rotated = torch.ormqr(qr, tau, noise, left=True, transpose=True)
        count = scales.numel()
        head = rotation@((rotation.T@rotated[:count])*scales[:, None])
        rotated = torch.cat((head, rotated[count:]), dim=0)
        noise = torch.ormqr(qr, tau, rotated, left=True, transpose=False)
    return noise/posterior.diagonal.sqrt()[:, None]


def draw_parameters(posterior, count, streams=None, *, generator=None):
    return unflatten_parameters(sample_vectors(posterior, count, streams, generator=generator),
                                posterior.layout)


def covariance_root(posterior, *, max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
    """Diagnostic S with S S.T = precision^-1, guarded before dense allocation."""
    from .dense import precision_matrix

    size = posterior.layout.numel
    guard_dense(size**2, posterior.mean.dtype, max_dense_bytes)
    identity = torch.eye(size, dtype=posterior.mean.dtype, device=posterior.mean.device)
    if posterior.representation == 'diagonal':
        return posterior.diagonal.rsqrt().diag()
    if posterior.representation == 'low_rank':
        return _low_rank_inverse_root(posterior, identity)
    if posterior.representation == 'dense':
        factor = posterior.factor
    else:
        factor, _ = cholesky_with_jitter(precision_matrix(posterior, max_dense_bytes=max_dense_bytes))
    return torch.linalg.solve_triangular(factor.T, identity, upper=True)
