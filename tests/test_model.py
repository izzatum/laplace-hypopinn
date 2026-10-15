import pytest
import torch
from torch import nn

from laplace_hypopinn.config import DEFAULT_WIDTHS
from laplace_hypopinn.model import (
    flatten_parameters,
    initialize_model,
    make_mlp_forward,
    module_adapter,
    predict,
    unflatten_parameters,
)
from laplace_hypopinn.types import RandomStreams
from reference.notebook_adapter import definitions


def test_historical_parameter_count_and_sequential_initialization():
    reference = definitions()["PINN"]
    streams = RandomStreams(4321)
    original = torch.random.get_rng_state().clone()
    with torch.random.fork_rng():
        torch.manual_seed(4321)
        state = torch.random.get_rng_state()
    for initialization in ("kaiming_normal", "xavier_normal", "kaiming_uniform", "xavier_uniform"):
        params = initialize_model(DEFAULT_WIDTHS, initialization, streams)
        assert sum(p.numel() for p in params.values()) == 2737
        with torch.random.fork_rng():
            torch.random.set_rng_state(state)
            module = reference(initialization)
            state = torch.random.get_rng_state()
        for actual, expected in zip(params.values(), module.parameters(), strict=True):
            assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), original)


def test_functional_forward_and_coordinate_gradients_match_reference():
    params = initialize_model(
        DEFAULT_WIDTHS, "kaiming_normal", RandomStreams(9), dtype=torch.float64
    )
    with torch.random.fork_rng():
        module = definitions()["PINN"]().double()
    with torch.no_grad():
        for target, value in zip(module.parameters(), params.values(), strict=True):
            target.copy_(value)
    coords = torch.tensor([[0.1, 0.2], [0.3, 1.2]], dtype=torch.float64, requires_grad=True)
    actual, expected = predict(params, coords), module(coords)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    ga = torch.autograd.grad(actual.sum(), coords, create_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), coords, create_graph=True)[0]
    torch.testing.assert_close(ga, ge, rtol=1e-11, atol=1e-11)
    assert coords.grad is None


@pytest.mark.parametrize("widths", [(2, 1), (2, 3, 5, 1), (2, 8, 8, 8, 8, 1)])
@pytest.mark.parametrize("activation", ["mish", "tanh", "silu"])
def test_configurable_architecture_and_flatten_roundtrip(widths, activation):
    params = initialize_model(widths, streams=RandomStreams(2), dtype=torch.float64)
    original = {k: v.clone() for k, v in params.items()}
    flat, layout = flatten_parameters(params)
    restored = unflatten_parameters(flat, layout)
    assert layout.numel == flat.numel()
    assert len(layout.blocks) == len(widths) - 1
    first = torch.cat((params["layers.0.weight"], params["layers.0.bias"][:, None]), dim=1)
    assert torch.equal(flat[: first.numel()], first.flatten())
    for name in params:
        assert torch.equal(params[name], restored[name])
        assert torch.equal(params[name], original[name])
    result = make_mlp_forward(activation)(restored, torch.ones(4, 2, dtype=torch.float64))
    assert result.shape == (4, 1)
    assert all(
        torch.isfinite(t).all() for t in torch.autograd.grad(result.sum(), tuple(params.values()))
    )


def test_custom_parameter_mapping_and_module_adapter():
    params = {
        "coefficient": torch.tensor(2.0, requires_grad=True),
        "offset": torch.tensor([1.0, 3.0], requires_grad=True),
    }
    vector, layout = flatten_parameters(params)
    rebuilt = unflatten_parameters(vector, layout)

    def forward(p, x):
        return p["coefficient"] * (x + p["offset"]).square().sum(-1, keepdim=True)

    coordinates = torch.ones(3, 2, requires_grad=True)
    assert forward(rebuilt, coordinates).shape == (3, 1)
    assert len(layout.blocks) == 2
    assert torch.equal(
        torch.autograd.grad(forward(rebuilt, coordinates).sum(), params["coefficient"])[0],
        torch.tensor(60.0),
    )
    with torch.random.fork_rng():
        module = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).eval()
    adapted_params, adapted_forward = module_adapter(module)
    torch.testing.assert_close(adapted_forward(adapted_params, coordinates), module(coordinates))
    assert (
        torch.autograd.grad(adapted_forward(adapted_params, coordinates).sum(), coordinates)[
            0
        ].shape
        == coordinates.shape
    )
    with pytest.raises(ValueError, match="eval"):
        module_adapter(module.train())


def test_layout_rejects_wrong_size_and_initialization_validation():
    params = initialize_model((2, 1), streams=RandomStreams(1))
    vector, layout = flatten_parameters(params)
    with pytest.raises(ValueError, match="size|length"):
        unflatten_parameters(vector[:-1], layout)
    with pytest.raises(ValueError, match="initialization"):
        initialize_model((2, 1), "unknown", RandomStreams(1))
    with pytest.raises(ValueError, match="activation"):
        make_mlp_forward("relu")


def test_custom_module_with_mlp_names_and_arbitrary_tensor_shapes_roundtrips():
    class Cubic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(8, dtype=torch.float64).reshape(2, 2, 2) / 10)
            self.bias = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

        def forward(self, x):
            return (torch.einsum("ni,nj,nk,ijk->n", x, x, x, self.weight) + self.bias)[:, None]

    class PointwiseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Cubic()])

        def forward(self, x):
            return self.layers[0](x)

    module = PointwiseModel().eval()
    parameters, forward = module_adapter(module)
    original = {name: value.clone() for name, value in parameters.items()}
    coordinates = torch.tensor([[0.2, 0.7], [0.5, 0.8]], dtype=torch.float64, requires_grad=True)
    expected = forward(parameters, coordinates)
    expected_gradients = torch.autograd.grad(expected.sum(), (*parameters.values(), coordinates))
    vector, layout = flatten_parameters(parameters)
    assert all(block.kind == "tensor" for block in layout.blocks)
    restored = unflatten_parameters(vector, layout)
    actual = forward(restored, coordinates)
    actual_gradients = torch.autograd.grad(actual.sum(), (*parameters.values(), coordinates))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    for name in parameters:
        assert torch.equal(parameters[name], restored[name])
        assert torch.equal(parameters[name], original[name])


def test_disconnected_linear_shapes_use_generic_layout():
    parameters = {
        "layers.0.weight": torch.ones(3, 2, dtype=torch.float64, requires_grad=True),
        "layers.0.bias": torch.ones(3, dtype=torch.float64, requires_grad=True),
        "layers.1.weight": torch.ones(1, 4, dtype=torch.float64, requires_grad=True),
        "layers.1.bias": torch.ones(1, dtype=torch.float64, requires_grad=True),
    }
    vector, layout = flatten_parameters(parameters)
    assert all(block.kind == "tensor" for block in layout.blocks)
    restored = unflatten_parameters(vector, layout)
    for name in parameters:
        assert torch.equal(restored[name], parameters[name])
    with pytest.raises(ValueError, match="MLP"):
        predict(parameters, torch.ones(2, 2, dtype=torch.float64))


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"layers.0.weight": torch.ones(1, 2, 2), "layers.0.bias": torch.tensor(0.0)},
        {
            "layers.0.weight": torch.ones(1, 2),
            "layers.0.bias": torch.ones(1),
            "extra": torch.ones(1),
        },
    ],
)
def test_default_mlp_predict_rejects_invalid_mapping(parameters):
    with pytest.raises(ValueError, match="MLP"):
        predict(parameters, torch.ones(2, 2))


def test_module_adapter_excludes_frozen_weights_from_training_and_jacobians():
    from laplace_hypopinn.config import TrainingConfig
    from laplace_hypopinn.training import train
    from laplace_hypopinn.types import TrainingBatch

    module = nn.Linear(2, 1, dtype=torch.float64).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.3, 0.4]], dtype=torch.float64))
        module.bias.fill_(0.2)
    module.weight.requires_grad_(False)
    original = {name: value.detach().clone() for name, value in module.named_parameters()}
    parameters, forward = module_adapter(module)
    assert tuple(parameters) == ("bias",)
    vector, layout = flatten_parameters(parameters)
    assert layout.numel == vector.numel() == 1
    coordinates = torch.tensor([[0.2, 0.7], [0.5, 0.8]], dtype=torch.float64)
    expected = module(coordinates).detach()
    torch.testing.assert_close(forward(parameters, coordinates), expected)
    jacobian = torch.func.jacrev(lambda p: forward(p, coordinates))(parameters)
    assert tuple(jacobian) == ("bias",)
    torch.testing.assert_close(jacobian["bias"], torch.ones(2, 1, 1, dtype=torch.float64))
    input_jacobian = torch.func.jacrev(lambda x: forward(parameters, x))(coordinates)
    torch.testing.assert_close(input_jacobian[0, 0, 0], original["weight"][0])
    batch = TrainingBatch(
        coordinates,
        torch.full((2,), 2.0, dtype=torch.float64),
        coordinates,
        torch.zeros(2, 1, dtype=torch.float64),
    )
    result = train(parameters, batch, TrainingConfig(steps=3), RandomStreams(1), forward=forward)
    assert tuple(result.parameters) == ("bias",)
    assert not torch.equal(result.parameters["bias"], parameters["bias"])
    trained_input_jacobian = torch.func.jacrev(lambda x: forward(result.parameters, x))(coordinates)
    torch.testing.assert_close(trained_input_jacobian, input_jacobian)
    for name, value in module.named_parameters():
        assert torch.equal(value, original[name])
    with torch.no_grad():
        module.weight.add_(100)
        module.bias.add_(100)
    torch.testing.assert_close(forward(parameters, coordinates), expected)


def test_module_adapter_rejects_all_frozen_models():
    module = nn.Linear(2, 1).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        module_adapter(module)


def test_module_adapter_isolates_frozen_constants_from_mutating_module_code():
    class MutatingFrozen(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.3, 0.4]), requires_grad=False)
            self.bias = nn.Parameter(torch.tensor(0.2))

        def forward(self, x):
            with torch.no_grad():
                self.weight.add_(1)
            return (x * self.weight).sum(-1) + self.bias

    module = MutatingFrozen().eval()
    parameters, forward = module_adapter(module)
    coordinates = torch.ones(2, 2)
    first = forward(parameters, coordinates)
    second = forward(parameters, coordinates)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(module.weight, torch.tensor([0.3, 0.4]))
