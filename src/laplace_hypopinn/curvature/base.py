"""Explicit posterior representations and guarded numerical primitives."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from ..model import predict
from ..types import Forward, ParameterLayout

DEFAULT_MAX_DENSE_BYTES = 256 * 1024 * 1024


def require_scalar(value, name, *, positive=False):
    if not math.isfinite(float(value)) or (float(value) <= 0 if positive else float(value) < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')


def guard_dense(numel, dtype, max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
    required = numel * torch.empty((), dtype=dtype).element_size()
    if max_dense_bytes < 0 or required > max_dense_bytes:
        raise MemoryError(f'dense allocation requires {required} bytes; limit is {max_dense_bytes}')


def cholesky_with_jitter(matrix):
    """Permit only numerical roundoff repair, and return the exact added jitter."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError('precision must be a finite square matrix')
    eps = torch.finfo(matrix.dtype).eps
    scale = max(float(matrix.abs().max()), torch.finfo(matrix.dtype).tiny)
    tolerance = 10 * matrix.shape[0] * eps * scale
    if not torch.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
        raise ValueError('precision must be symmetric')
    matrix = (matrix + matrix.T) / 2
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info) == 0:
        return factor, 0.
    minimum = float(torch.linalg.eigvalsh(matrix).min())
    if minimum < -tolerance:
        raise ValueError('precision is substantially indefinite')
    jitter = max(eps*scale, -minimum + eps*scale)
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    for _ in range(3):
        factor, info = torch.linalg.cholesky_ex(matrix + jitter*eye)
        if int(info) == 0:
            return factor, jitter
        jitter *= 10
        if jitter > 10*tolerance:
            break
    raise ValueError('precision could not be factored with roundoff jitter')


@dataclass(frozen=True)
class Posterior:
    """Gaussian mean and a tagged precision representation in canonical layout.

    ``blocks`` and ``factor`` contain lower Cholesky factors of precision;
    ``low_rank`` contains columns B for precision diag(diagonal) + B B.T.
    ``eigenbasis`` stores (output basis, augmented-input basis, precision eigenvalues).
    """

    mean: Tensor
    layout: ParameterLayout
    representation: str
    diagonal: Tensor | None = None
    blocks: tuple[Tensor, ...] = ()
    factor: Tensor | None = None
    low_rank: Tensor | None = None
    eigenbasis: tuple[tuple[Tensor, Tensor, Tensor], ...] = ()
    metadata: Mapping = field(default_factory=dict)
    forward: Forward = predict
    prediction_kind: str = 'travel_time'

    def __post_init__(self):
        if self.mean.shape != (self.layout.numel,) or not torch.isfinite(self.mean).all():
            raise ValueError('posterior mean must be finite and match layout')
        def check(value, shape, positive=False):
            if value is None or value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('invalid posterior precision shape or nonfinite values')
            if value.dtype != self.mean.dtype or value.device != self.mean.device:
                raise ValueError('posterior precision and mean dtype/device must match')
            if positive and not (value > 0).all():
                raise ValueError('posterior precision must be positive')
        def factor(value, size):
            check(value, (size, size))
            if not (value.diag() > 0).all() or not torch.equal(value, value.tril()):
                raise ValueError('precision factor must be lower triangular with positive diagonal')
        if self.representation == 'diagonal':
            check(self.diagonal, self.mean.shape, True)
        elif self.representation == 'dense':
            factor(self.factor, self.layout.numel)
        elif self.representation == 'block_diagonal':
            if len(self.blocks) != len(self.layout.blocks):
                raise ValueError('precision blocks must match parameter layout')
            for value, block in zip(self.blocks, self.layout.blocks, strict=True):
                factor(value, block.numel)
        elif self.representation == 'low_rank':
            check(self.diagonal, self.mean.shape, True)
            if self.low_rank is None or self.low_rank.ndim != 2:
                raise ValueError('low_rank must be a matrix')
            check(self.low_rank, (self.layout.numel, self.low_rank.shape[1]))
        elif self.representation == 'kronecker':
            if len(self.eigenbasis) != len(self.layout.blocks):
                raise ValueError('eigenbasis blocks must match parameter layout')
            for (qg, qa, values), block in zip(self.eigenbasis, self.layout.blocks, strict=True):
                if len(block.shape) != 2:
                    raise ValueError('Kronecker blocks require matrix parameter layout')
                rows, columns = block.shape
                check(qg, (rows, rows))
                check(qa, (columns, columns))
                check(values, block.shape, True)
                for basis in (qg, qa):
                    identity = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device)
                    if not torch.allclose(basis.T@basis, identity, rtol=1e-5, atol=1e-6):
                        raise ValueError('Kronecker eigenbasis must be orthonormal')
        else:
            raise ValueError(f'unknown posterior representation {self.representation!r}')

    @property
    def meanflat(self):
        return self.mean
