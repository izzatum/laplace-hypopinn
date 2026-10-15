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
        diagonal, columns = posterior.diagonal, posterior.low_rank
        rank = columns.shape[1]
        if rank:
            scaled = columns/diagonal[:, None]
            middle = torch.eye(rank, dtype=mean.dtype, device=mean.device)+columns.T@scaled
            factor, _ = cholesky_with_jitter(middle)
        samples = []
        for _ in range(count):
            noise = normal((mean.numel(),))*diagonal.sqrt()
            if rank:
                noise = noise+columns@normal((rank,))
                right = columns.T@(noise/diagonal)
                solved = torch.cholesky_solve(right[:, None], factor).reshape(-1)
                samples.append(noise/diagonal-scaled@solved)
            else:
                samples.append(noise/diagonal)
        offsets = torch.stack(samples)
    else:
        raise ValueError(f'unsupported posterior representation {kind!r}')
    return (offsets+mean).detach()


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
    if posterior.representation == 'dense':
        factor = posterior.factor
    else:
        factor, _ = cholesky_with_jitter(precision_matrix(posterior, max_dense_bytes=max_dense_bytes))
    return torch.linalg.solve_triangular(factor.T, identity, upper=True)
