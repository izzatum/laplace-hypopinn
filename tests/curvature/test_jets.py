import pytest
import torch

from laplace_hypopinn.model import initialize_model, make_mlp_forward, flatten_parameters
from laplace_hypopinn.types import RandomStreams
from laplace_hypopinn.curvature.residuals import ResidualBatch, residual_values
from laplace_hypopinn.curvature.jacobians import iter_jacobians
from laplace_hypopinn.curvature.jets import jet_forward, residual_layer_terms


def parameters(widths=(2, 3, 2, 1)):
    return initialize_model(widths, streams=RandomStreams(49), dtype=torch.float64)


@pytest.mark.parametrize('activation', ['mish', 'tanh', 'silu'])
@pytest.mark.parametrize('family,count', [('pde', 5), ('data', 2), ('positivity', 4)])
def test_channel_outer_products_reconstruct_complete_mixed_jacobian(activation, family, count):
    params = parameters()
    params['layers.2.bias'] = torch.tensor([-2.], dtype=torch.float64, requires_grad=True)
    forward = make_mlp_forward(activation)
    coordinates = torch.linspace(-.6, .8, count * 2, dtype=torch.float64).reshape(count, 2)
    batch = ResidualBatch(family, coordinates, torch.full((count,), 2., dtype=torch.float64), torch.ones(count, dtype=torch.float64))
    terms = residual_layer_terms(params, batch, forward=forward)
    exact = torch.cat(tuple(iter_jacobians(params, (batch,), 2, forward=forward)))
    _, layout = flatten_parameters(params)
    for layer, block in zip(terms.layers, layout.blocks):
        assert layer.activations.shape[1] == (3 if family == 'pde' else 1)
        reconstructed = torch.einsum('nso,nsi->noi', layer.adjoints, layer.activations)
        torch.testing.assert_close(reconstructed.flatten(1), exact[:, block.slice], atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(terms.residuals, residual_values(params, batch, forward=forward))


def test_jet_values_and_coordinate_derivatives():
    params = parameters((2, 5, 1))
    coords = torch.tensor([[.1, .4], [-.2, .7]], dtype=torch.float64)
    forward = make_mlp_forward('silu')
    trace = jet_forward(params, coords, forward=forward)
    expected = torch.func.vmap(torch.func.jacrev(lambda x: forward(params, x[None]).reshape(())))(coords)
    torch.testing.assert_close(trace.output[:, 0, :], forward(params, coords))
    torch.testing.assert_close(trace.output[:, 1:, 0], expected)
    for layer in trace.layers:
        torch.testing.assert_close(layer.activations[:, 0, -1], torch.ones(len(coords), dtype=coords.dtype))
        assert torch.count_nonzero(layer.activations[:, 1:, -1]) == 0


def test_custom_forward_requires_explicit_compatible_jet_adapter():
    with pytest.raises(ValueError, match='jet.adapter'):
        jet_forward(parameters(), torch.zeros(2, 2, dtype=torch.float64), forward=lambda p, x: x[:, :1])


def test_explicit_custom_affine_jet_adapter_supports_named_matrix_parameters():
    from laplace_hypopinn.curvature.jets import JetLayer, JetTrace

    params = {'coefficient': torch.tensor([[.3, -.7, .2]], dtype=torch.float64)}
    coords = torch.tensor([[.2, .1], [.7, -.3]], dtype=torch.float64)

    def forward(parameters, coordinates):
        augmented = torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), dim=1)
        return augmented @ parameters['coefficient'].T

    def adapter(parameters, coordinates):
        value_inputs = torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), dim=1)
        z_inputs = torch.tensor([1., 0., 0.], dtype=coordinates.dtype).expand_as(value_inputs)
        x_inputs = torch.tensor([0., 1., 0.], dtype=coordinates.dtype).expand_as(value_inputs)
        inputs = torch.stack((value_inputs, z_inputs, x_inputs), dim=1)
        pre = tuple(inputs[:, channel] @ parameters['coefficient'].T for channel in range(3))
        return JetTrace((JetLayer(inputs, pre),), torch.stack(pre, dim=1))

    batch = ResidualBatch('pde', coords, torch.ones(2, dtype=coords.dtype), torch.ones(2, dtype=coords.dtype))
    terms = residual_layer_terms(params, batch, forward=forward, jet_adapter=adapter)
    exact = torch.cat(tuple(iter_jacobians(params, (batch,), 1, forward=forward)))
    torch.testing.assert_close(terms.layers[0].matrices.flatten(1), exact)
