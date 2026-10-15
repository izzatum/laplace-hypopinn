"""First-order coordinate jets and exact local residual Jacobian terms.

No hooks or mutable model state are used. Every layer keeps its value/z/x
preactivations in the same graph, so a reverse pass through the residual sum
includes the mixed coordinate/parameter derivatives of the eikonal residual.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from ..model import flatten_parameters, predict


@dataclass(frozen=True)
class JetLayer:
    """Bias-augmented inputs and value/z/x preactivations of one affine layer."""

    activations: Tensor
    preactivations: tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class JetTrace:
    layers: tuple[JetLayer, ...]
    output: Tensor  # (point, value/z/x, scalar output)


@dataclass(frozen=True)
class LayerTerm:
    activations: Tensor  # (point, active channel, input including bias)
    adjoints: Tensor  # (point, active channel, output)

    @property
    def matrices(self):
        """Exact dr_i/d[W|b] in the canonical row-major layer layout."""
        return torch.einsum('nso,nsi->noi', self.adjoints, self.activations)


@dataclass(frozen=True)
class LayerTerms:
    layers: tuple[LayerTerm, ...]
    residuals: Tensor


JetAdapter = Callable[[dict[str, Tensor], Tensor], JetTrace]


def _activate(value, kind):
    if kind == 'tanh':
        output = torch.tanh(value)
        return output, 1 - output.square()
    sigmoid = torch.sigmoid(value)
    if kind == 'silu':
        return F.silu(value), sigmoid * (1 + value * (1 - sigmoid))
    if kind == 'mish':
        softplus_tanh = torch.tanh(F.softplus(value))
        return F.mish(value), softplus_tanh + value * sigmoid * (1 - softplus_tanh.square())
    raise ValueError(f'unsupported jet activation {kind!r}')


def jet_forward(parameters, coordinates, *, forward=predict, jet_adapter=None) -> JetTrace:
    """Evaluate pointwise scalar MLP jets with any supported depth and width.

    Custom models must supply ``jet_adapter(parameters, coordinates)`` returning
    a JetTrace whose affine layers agree with the canonical parameter blocks.
    Merely supplying an arbitrary forward never silently selects the MLP path.
    The adapter must preserve all preactivation graph dependencies and represent
    every parameter block as an augmented affine matrix.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError('jet coordinates must have shape (n,2)')
    # Local leaves make jets available even for frozen parameter snapshots.
    local = {name: value.detach().clone().requires_grad_(True) for name, value in parameters.items()}
    if jet_adapter is not None:
        trace = jet_adapter(local, coordinates)
        if not isinstance(trace, JetTrace):
            raise ValueError('jet_adapter must return a compatible JetTrace')
        return trace
    if getattr(forward, 'model_kind', None) != 'mlp':
        raise ValueError('custom forward requires an explicit compatible jet_adapter')
    _, layout = flatten_parameters(local)
    if any(block.kind != 'linear' for block in layout.blocks):
        raise ValueError('MLP jet parameters must use paired layers.i.weight/layers.i.bias names')
    activation = getattr(forward, 'activation', None)
    hidden = (
        coordinates,
        torch.stack((torch.ones_like(coordinates[:, 0]), torch.zeros_like(coordinates[:, 1])), dim=-1),
        torch.stack((torch.zeros_like(coordinates[:, 0]), torch.ones_like(coordinates[:, 1])), dim=-1),
    )
    layers = []
    for index, block in enumerate(layout.blocks):
        weight, bias = (local[name] for name in block.parameter_names)
        one = torch.ones_like(hidden[0][:, :1])
        zero = torch.zeros_like(one)
        inputs = torch.stack(tuple(torch.cat((value, one if channel == 0 else zero), dim=-1) for channel, value in enumerate(hidden)), dim=1)
        pre = tuple(F.linear(value, weight, bias if channel == 0 else None) for channel, value in enumerate(hidden))
        layers.append(JetLayer(inputs, pre))
        if index < len(layout.blocks) - 1:
            value, derivative = _activate(pre[0], activation)
            hidden = (value, derivative * pre[1], derivative * pre[2])
        else:
            hidden = pre
    if hidden[0].shape[1] != 1:
        raise ValueError('jet model must return one scalar per point')
    return JetTrace(tuple(layers), torch.stack(hidden, dim=1))


def residual_layer_terms(parameters, residual_batch, *, forward=predict, jet_adapter=None) -> LayerTerms:
    """Reconstruct complete per-row residual Jacobians as channel outer sums.

    Data and positivity residuals expose only the value channel; PDE residuals
    expose all three. Bias inputs are one for values and zero for derivatives.
    Returned statistics are detached, bounded by the caller's residual chunk.
    """
    with torch.enable_grad():
        trace = jet_forward(parameters, residual_batch.coordinates, forward=forward, jet_adapter=jet_adapter)
        output = trace.output
        if residual_batch.family == 'pde':
            residuals = output[:, 1:, 0].square().sum(dim=1) - residual_batch.targets.reshape(-1).square().reciprocal()
            channels = 3
        elif residual_batch.family == 'data':
            residuals = output[:, 0, 0] - residual_batch.targets.reshape(-1)
            channels = 1
        elif residual_batch.family == 'positivity':
            residuals = 2 * torch.relu(-output[:, 0, 0])
            channels = 1
        else:
            raise ValueError('unknown residual family')
        nodes = tuple(value for layer in trace.layers for value in layer.preactivations[:channels])
        derivatives = torch.autograd.grad(residuals.sum(), nodes, allow_unused=True)
        layers = []
        for index, layer in enumerate(trace.layers):
            adjoints = tuple(
                torch.zeros_like(layer.preactivations[channel]) if derivatives[index * channels + channel] is None else derivatives[index * channels + channel]
                for channel in range(channels)
            )
            layers.append(LayerTerm(layer.activations[:, :channels].detach(), torch.stack(adjoints, dim=1).detach()))
        return LayerTerms(tuple(layers), residuals.detach())
