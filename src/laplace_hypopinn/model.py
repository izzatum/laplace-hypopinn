"""Pure forward evaluation, explicit initialization and differentiable vectorization."""

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    ACTIVATIONS,
    DEFAULT_WIDTHS,
    INITIALIZATIONS,
    resolve_device,
    resolve_dtype,
    validate_widths,
)
from .types import Forward, ParameterBlock, ParameterLayout, Parameters, RandomStreams


def initialize_model(
    widths=DEFAULT_WIDTHS,
    initialization="kaiming_normal",
    streams=None,
    *,
    dtype=torch.float32,
    device="cpu",
) -> Parameters:
    """Match all default Linear draws before notebook-style weight reinitialization."""
    widths = validate_widths(widths)
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"Unknown initialization {initialization!r}")
    device, dtype = resolve_device(device), resolve_dtype(dtype)
    streams = RandomStreams(device=device) if streams is None else streams
    generator = streams.torch if hasattr(streams, "torch") else streams
    if torch.device(generator.device) != device:
        raise ValueError("initialization generator and requested device must match")
    parameters = {}
    for i, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        weight = torch.empty((outputs, inputs), dtype=dtype, device=device)
        bias = torch.empty(outputs, dtype=dtype, device=device)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
        nn.init.uniform_(bias, -1 / math.sqrt(inputs), 1 / math.sqrt(inputs), generator=generator)
        parameters[f"layers.{i}.weight"] = weight
        parameters[f"layers.{i}.bias"] = bias
    initializer = getattr(nn.init, initialization + "_")
    for i in range(len(widths) - 1):
        initializer(parameters[f"layers.{i}.weight"], generator=generator)
        parameters[f"layers.{i}.bias"].zero_()
    return {name: value.requires_grad_() for name, value in parameters.items()}


def _has_connected_linear_shapes(parameters):
    """Recognize linear layer structure without assigning semantics to names alone."""
    if not parameters or len(parameters) % 2:
        return False
    expected_names = tuple(
        name
        for i in range(len(parameters) // 2)
        for name in (f"layers.{i}.weight", f"layers.{i}.bias")
    )
    if set(parameters) != set(expected_names):
        return False
    previous_outputs = None
    for i in range(len(parameters) // 2):
        weight, bias = parameters[f"layers.{i}.weight"], parameters[f"layers.{i}.bias"]
        if weight.ndim != 2 or bias.shape != (weight.shape[0],) or min(weight.shape) <= 0:
            return False
        if previous_outputs is not None and weight.shape[1] != previous_outputs:
            return False
        previous_outputs = weight.shape[0]
    return True


def _mlp_predict(parameters, coordinates, activation):
    if not _has_connected_linear_shapes(parameters):
        raise ValueError("MLP parameters require complete connected linear weight/bias pairs")
    if (
        parameters["layers.0.weight"].shape[1] != 2
        or parameters[f"layers.{len(parameters) // 2 - 1}.weight"].shape[0] != 1
    ):
        raise ValueError("MLP must map two coordinates to one scalar")
    hidden = coordinates
    count = len(parameters) // 2
    for i in range(count):
        hidden = F.linear(hidden, parameters[f"layers.{i}.weight"], parameters[f"layers.{i}.bias"])
        if i < count - 1:
            hidden = {"mish": F.mish, "tanh": torch.tanh, "silu": F.silu}[activation](hidden)
    return hidden


def predict(parameters: Mapping[str, Tensor], coordinates: Tensor) -> Tensor:
    """Default Mish MLP, with depth and widths inferred from the parameter mapping."""
    return _mlp_predict(parameters, coordinates, "mish")


predict.activation = "mish"
predict.model_kind = "mlp"


def make_mlp_forward(activation="mish") -> Forward:
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")

    def forward(parameters, coordinates):
        return _mlp_predict(parameters, coordinates, activation)

    forward.activation = activation
    forward.model_kind = "mlp"
    return forward


def module_adapter(module: nn.Module) -> tuple[Parameters, Forward]:
    """Snapshot an eval module and buffers for torch.func-compatible scalar models.

    The caller is responsible for a differentiable, deterministic, pointwise model.
    Output shape may be (n,) or (n,1). Parameters and module buffers are isolated
    from the original module, including custom modules that mutate their buffers.
    """
    import copy

    if module.training or any(child.training for child in module.modules()):
        raise ValueError("module_adapter requires every module to be in eval mode")
    local_module = copy.deepcopy(module)
    parameters = {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in local_module.named_parameters()
    }
    buffers = {name: value.detach().clone() for name, value in local_module.named_buffers()}

    def forward(parameters, coordinates):
        return torch.func.functional_call(
            local_module,
            (parameters, {k: v.clone() for k, v in buffers.items()}),
            (coordinates,),
            strict=True,
        )

    return parameters, forward


def flatten_parameters(parameters: Mapping[str, Tensor]) -> tuple[Tensor, ParameterLayout]:
    """Flatten MLP [W|b] row-major blocks, arbitrary named tensors in insertion order."""
    if not parameters:
        raise ValueError("parameter mapping must be nonempty")
    names = tuple(parameters)
    pieces, blocks = [], []
    mlp = _has_connected_linear_shapes(parameters) and names == tuple(
        name for i in range(len(names) // 2) for name in (f"layers.{i}.weight", f"layers.{i}.bias")
    )
    start = 0
    if mlp:
        for i in range(len(names) // 2):
            weight_name, bias_name = f"layers.{i}.weight", f"layers.{i}.bias"
            weight, bias = parameters[weight_name], parameters[bias_name]
            piece = torch.cat((weight, bias[:, None]), dim=1)
            stop = start + piece.numel()
            blocks.append(
                ParameterBlock(
                    f"layers.{i}",
                    start,
                    stop,
                    tuple(piece.shape),
                    (weight_name, bias_name),
                    (tuple(weight.shape), tuple(bias.shape)),
                    "linear",
                )
            )
            pieces.append(piece.reshape(-1))
            start = stop
    else:
        for name, value in parameters.items():
            stop = start + value.numel()
            blocks.append(
                ParameterBlock(
                    name, start, stop, tuple(value.shape), (name,), (tuple(value.shape),)
                )
            )
            pieces.append(value.reshape(-1))
            start = stop
    return torch.cat(pieces), ParameterLayout(tuple(blocks), names, start)


def unflatten_parameters(vector: Tensor, layout: ParameterLayout) -> Parameters:
    """Return differentiable views; supports an arbitrary leading sample shape."""
    if vector.ndim < 1 or vector.shape[-1] != layout.numel:
        raise ValueError(f"vector last dimension must have length {layout.numel}")
    leading = vector.shape[:-1]
    parameters = {}
    for block in layout.blocks:
        value = vector[..., block.start : block.stop].reshape(tuple(leading) + block.shape)
        if block.kind == "linear":
            parameters[block.parameter_names[0]] = value[..., :-1]
            parameters[block.parameter_names[1]] = value[..., -1]
        else:
            parameters[block.parameter_names[0]] = value
    return {name: parameters[name] for name in layout.names}
