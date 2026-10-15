"""Loss contracts tested against the notebook equations, without notebook execution."""

import pytest
import torch

from laplace_hypopinn.config import TrainingConfig
from laplace_hypopinn.types import TrainingBatch


def polynomial(parameters, coordinates):
    z, x = coordinates.unbind(-1)
    return (
        parameters["weights"][0] * z.square() + parameters["weights"][1] * x + parameters["offset"]
    )


def example(dtype=torch.float64):
    parameters = {
        "weights": torch.tensor([0.4, -0.3], dtype=dtype, requires_grad=True),
        "offset": torch.tensor(-0.2, dtype=dtype, requires_grad=True),
    }
    coordinates = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 0.2]], dtype=dtype)
    receivers = torch.tensor([[0.1, 0.2], [0.8, 0.6]], dtype=dtype)
    batch = TrainingBatch(
        coordinates,
        torch.tensor([2.0, 3.0, 2.5], dtype=dtype),
        receivers,
        torch.tensor([[0.2], [0.5]], dtype=dtype),
    )
    return parameters, batch


@pytest.mark.parametrize("precision", [0.0, 0.7])
def test_loss_components_and_gradients_match_notebook_equations(precision):
    from laplace_hypopinn.losses import loss_terms

    parameters, batch = example()
    config = TrainingConfig(
        pde_weight=1.3, data_weight=0.4, positivity_weight=0.8, prior_precision=precision
    )
    coordinates = batch.coordinates.clone().requires_grad_()
    times = polynomial(parameters, coordinates)
    gradient = torch.autograd.grad(times.sum(), coordinates, create_graph=True)[0]
    pde = (gradient.square().sum(-1) - batch.velocity.pow(-2)).square().mean()
    data = (
        (polynomial(parameters, batch.receiver_coordinates) - batch.receiver_times.reshape(-1))
        .square()
        .mean()
    )
    positivity = ((1 - torch.sign(times)) * times.abs()).square().mean()
    prior = 0.5 * precision * sum(value.square().sum() for value in parameters.values())
    total = 1.3 * pde + 0.4 * data + 0.8 * positivity + prior
    actual = loss_terms(parameters, batch, config, forward=polynomial)
    for name, expected in [
        ("pde", pde),
        ("data", data),
        ("positivity", positivity),
        ("prior", prior),
        ("total", total),
    ]:
        torch.testing.assert_close(getattr(actual, name), expected)
    for name, expected in [
        ("pde", pde),
        ("data", data),
        ("positivity", positivity),
        ("prior", prior),
        ("total", total),
    ]:
        expected_grad = torch.autograd.grad(
            expected, tuple(parameters.values()), retain_graph=True, materialize_grads=True
        )
        actual_grad = torch.autograd.grad(
            getattr(actual, name),
            tuple(parameters.values()),
            retain_graph=True,
            materialize_grads=True,
        )
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value)


def test_residual_weights_reconstruct_loss_and_prior_is_global():
    from laplace_hypopinn.losses import loss_terms
    from laplace_hypopinn.curvature.residuals import make_residual_plan, residual_values

    parameters, batch = example()
    config = TrainingConfig(
        pde_weight=0.8, data_weight=1.7, positivity_weight=0.3, prior_precision=2.0
    )
    plan = make_residual_plan(batch, config)
    assert isinstance(plan, tuple)
    assert [entry.family for entry in plan] == ["data", "pde", "positivity"]
    for entry, count, weight in zip(plan, [2, 3, 3], [1.7, 0.8, 0.3]):
        torch.testing.assert_close(
            entry.alpha, torch.full((count,), 2 * weight / count, dtype=torch.float64)
        )
    reconstructed = sum(
        0.5 * (entry.alpha * residual_values(parameters, entry, forward=polynomial).square()).sum()
        for entry in plan
    )
    actual = loss_terms(parameters, batch, config, forward=polynomial)
    torch.testing.assert_close(reconstructed + actual.prior, actual.total)
    assert all(entry.family != "prior" for entry in plan)
    changed_prior = make_residual_plan(
        batch,
        TrainingConfig(pde_weight=0.8, data_weight=1.7, positivity_weight=0.3, prior_precision=9.0),
    )
    assert all(torch.equal(a.alpha, b.alpha) for a, b in zip(plan, changed_prior))


def test_zero_weight_skip_fixed_selection_and_zero_positivity_gradient():
    from laplace_hypopinn.losses import loss_terms
    from laplace_hypopinn.curvature.residuals import make_residual_plan

    parameters, batch = example()
    config = TrainingConfig(pde_weight=0.0, data_weight=1.0, positivity_weight=0.0)
    plan = make_residual_plan(batch, config)
    assert len(plan) == 1 and plan[0].family == "data"
    copied = plan[0].coordinates.clone()
    batch.receiver_coordinates.add_(1.0)
    torch.testing.assert_close(plan[0].coordinates, copied)
    parameters = {
        "weights": torch.zeros(2, dtype=torch.float64, requires_grad=True),
        "offset": torch.zeros((), dtype=torch.float64, requires_grad=True),
    }
    result = loss_terms(parameters, batch, TrainingConfig(), forward=polynomial)
    grads = torch.autograd.grad(result.positivity, tuple(parameters.values()))
    assert result.positivity.item() == 0.0
    assert all(torch.count_nonzero(gradient).item() == 0 for gradient in grads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loss_preserves_dtype_and_global_rng(dtype):
    from laplace_hypopinn.losses import loss_terms

    parameters, batch = example(dtype)
    state = torch.random.get_rng_state().clone()
    snapshots = {name: value.clone() for name, value in parameters.items()}
    result = loss_terms(parameters, batch, TrainingConfig(), forward=polynomial)
    assert result.total.dtype == dtype
    assert torch.equal(state, torch.random.get_rng_state())
    for name, value in parameters.items():
        torch.testing.assert_close(value, snapshots[name])


def test_observed_only_problem_needs_no_reference_or_true_source():
    from laplace_hypopinn.curvature.residuals import make_residual_plan
    from laplace_hypopinn.losses import loss_terms
    from laplace_hypopinn.types import Grid, Problem

    parameters, batch = example()
    axis = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    grid = Grid(axis, axis, torch.full((3, 3), 2.0, dtype=torch.float64))
    problem = Problem(grid, batch.receiver_coordinates, batch.receiver_times)
    observed = TrainingBatch(
        batch.coordinates, batch.velocity, problem.receiver_coordinates, problem.receiver_times
    )
    assert problem.reference_times is None and problem.true_source is None
    terms = loss_terms(parameters, observed, TrainingConfig(), forward=polynomial)
    assert torch.isfinite(terms.total)
    assert (
        sum(entry.coordinates.shape[0] for entry in make_residual_plan(observed, TrainingConfig()))
        == 8
    )


@pytest.mark.parametrize('requires_grad', [False, True])
def test_custom_coordinate_independent_model_has_zero_coordinate_gradient(requires_grad):
    from laplace_hypopinn.losses import loss_terms

    _, batch = example()
    parameters = {'bias': torch.tensor(.2, dtype=torch.float64, requires_grad=requires_grad)}
    def constant(parameters, coordinates):
        return parameters['bias'].expand(coordinates.shape[0])
    terms = loss_terms(parameters, batch, TrainingConfig(), forward=constant)
    torch.testing.assert_close(terms.pde, batch.velocity.pow(-4).mean())
    if requires_grad:
        gradient = torch.autograd.grad(terms.total, parameters['bias'])[0]
        torch.testing.assert_close(gradient, 2*(parameters['bias']-batch.receiver_times.mean()))
