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
