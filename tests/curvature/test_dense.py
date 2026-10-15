import pytest
import torch

from laplace_hypopinn.config import TrainingConfig
from laplace_hypopinn.model import flatten_parameters
from test_losses import example, polynomial


def reference():
    from laplace_hypopinn.curvature.residuals import make_residual_plan
    from laplace_hypopinn.curvature.jacobians import iter_jacobians
    parameters, batch = example()
    plan = make_residual_plan(batch, TrainingConfig(pde_weight=.7, data_weight=1.2))
    jacobian = torch.cat(list(iter_jacobians(parameters, plan, 2, forward=polynomial)))
    return parameters, plan, jacobian


@pytest.mark.parametrize('structure', ['full', 'block_diagonal', 'diagonal'])
def test_streaming_ggn_matches_dense_oracle(structure):
    from laplace_hypopinn.curvature.dense import fit_ggn, precision_matrix
    parameters, plan, jacobian = reference()
    _, layout = flatten_parameters(parameters)
    expected = .8*jacobian.T@jacobian + 1.3*torch.eye(layout.numel, dtype=jacobian.dtype)
    if structure == 'diagonal':
        expected = expected.diag().diag()
    elif structure == 'block_diagonal':
        expected = torch.block_diag(*(expected[b.slice, b.slice] for b in layout.blocks))
    for chunk in [1, 2, 100]:
        posterior = fit_ggn(parameters, plan, structure=structure, prior_precision=1.3,
                            scale=.8, chunk_size=chunk, forward=polynomial)
        torch.testing.assert_close(precision_matrix(posterior), expected)
        assert posterior.metadata['estimator'] == 'weighted_residual_ggn'


def test_rank_deficient_data_with_positive_prior_is_valid():
    from laplace_hypopinn.curvature.dense import fit_ggn, precision_matrix
    parameters, _, _ = reference()
    posterior = fit_ggn(parameters, (), structure='full', prior_precision=.2)
    torch.testing.assert_close(precision_matrix(posterior), .2*torch.eye(3, dtype=torch.float64))


@pytest.mark.parametrize('kwargs', [{'scale': -1}, {'scale': float('nan')},
                                    {'prior_precision': 0}, {'prior_precision': float('inf')}])
def test_invalid_precision_and_scale_rejected(kwargs):
    from laplace_hypopinn.curvature.dense import fit_ggn
    parameters, plan, _ = reference()
    with pytest.raises(ValueError):
        fit_ggn(parameters, plan, **kwargs, forward=polynomial)


def test_dense_memory_guards_apply_before_accumulation_and_materialization():
    from laplace_hypopinn.curvature.dense import fit_ggn, precision_matrix
    parameters, plan, _ = reference()
    with pytest.raises(MemoryError):
        fit_ggn(parameters, plan, structure='full', max_dense_bytes=1, forward=polynomial)
    posterior = fit_ggn(parameters, plan, structure='diagonal', forward=polynomial)
    with pytest.raises(MemoryError):
        precision_matrix(posterior, max_dense_bytes=1)


def test_roundoff_jitter_is_recorded_and_indefiniteness_rejected():
    from laplace_hypopinn.curvature.base import cholesky_with_jitter
    matrix = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float64)
    factor, jitter = cholesky_with_jitter(matrix)
    assert 0 < jitter < 1e-12
    torch.testing.assert_close(factor@factor.T, matrix+jitter*torch.eye(2, dtype=matrix.dtype))
    with pytest.raises(ValueError, match='indefinite'):
        cholesky_with_jitter(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64))
