from dataclasses import replace

import numpy as np
import pytest
import torch

from laplace_hypopinn.config import DataConfig, ExperimentConfig, LaplaceConfig, ModelConfig, TrainingConfig
from laplace_hypopinn.data import prepare_problem
from laplace_hypopinn.experiment import run_experiment
from laplace_hypopinn.presets import load_preset


def tiny(method='diagonal', name='tiny'):
    return ExperimentConfig(name=name, model=ModelConfig(widths=(2, 2, 1)),
        data=DataConfig(collocation_count=4, grid_shape=(4, 5), receiver_count=3),
        training=TrainingConfig(steps=1), laplace=LaplaceConfig(method=method, batches=1,
        batch_size=4, samples=2, chunk_size=3, rank=2, inducing_count=2, variational_steps=1))


@pytest.mark.parametrize('method', ['legacy_diagonal', 'diagonal', 'block_diagonal', 'full', 'kfac', 'efb', 'inf', 'valla'])
def test_all_methods(method):
    result = run_experiment(tiny(method))
    assert result.summary.locations.shape == (2, 2)
    assert result.metadata['method'] == method
    assert torch.isfinite(result.map_field).all()


@pytest.mark.parametrize('name', ['ex1', 'ex2', 'ex3', 'ex3_2', 'ex3_3'])
def test_preset_short_runs(name):
    preset = load_preset(name)
    config = tiny(name=name)
    # Explicit supplied problem permits smoke testing every historical preset
    # without synthesizing Otway reference data or importing fast marching.
    problem = prepare_problem(config)
    result = run_experiment(replace(preset, model=config.model, data=config.data,
        training=config.training, laplace=config.laplace), problem=problem)
    assert result.map_field.shape == problem.grid.shape
    if name == 'ex1':
        assert len(result.runs) == 4
        assert result.posterior is None
        assert len({r.metadata['initial_parameter_hash'] for r in result.runs}) == 4
        assert len({r.metadata['collocation_hash'] for r in result.runs}) == 1


def test_custom_observed_model_artifacts_and_rng(tmp_path):
    config = tiny('full')
    problem = prepare_problem(config)
    problem = replace(problem, true_source=None, reference_times=None)
    p = {'coefficients': torch.tensor([.2, .3, .4])}
    def custom(p, x):
        return x @ p['coefficients'][:2] + p['coefficients'][2]
    original = p['coefficients'].clone()
    rng = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    result = run_experiment(config, problem=problem, parameters=p, forward=custom, output=tmp_path/'run')
    torch.testing.assert_close(p['coefficients'], original)
    assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_equal(numpy_state, np.random.get_state())
    assert (tmp_path/'run'/'config.json').exists()
    assert (tmp_path/'run'/'metrics.json').exists()
    assert (tmp_path/'run'/'checkpoint.pt').exists()
    with np.load(tmp_path/'run'/'arrays.npz') as arrays:
        assert arrays['posterior_locations'].shape == (2, 2)
        assert 'true_source' not in arrays
    with pytest.raises(FileExistsError):
        run_experiment(config, output=tmp_path/'run')
    assert result.problem.true_source is None


def test_custom_model_jet_rejection():
    config = tiny('kfac')
    p = {'vector': torch.tensor([.3, .4])}
    with pytest.raises((ValueError, NotImplementedError), match='jet|MLP|adapter|model'):
        run_experiment(config, parameters=p, forward=lambda p, x: x@p['vector'])


@pytest.mark.parametrize('activation', ['tanh', 'silu'])
def test_free_architecture_and_activation_metadata(activation):
    config = tiny('kfac')
    config = replace(config, model=replace(config.model, widths=(2, 3, 2, 3, 1), activation=activation))
    result = run_experiment(config)
    assert result.metadata['forward']['activation'] == activation
    assert len(result.posterior.layout.blocks) == 4


def test_ex1_initialization_matches_sequential_stream():
    from laplace_hypopinn.config import INITIALIZATIONS
    from laplace_hypopinn.experiment import _hash_tensors
    from laplace_hypopinn.model import initialize_model
    from laplace_hypopinn.types import RandomStreams
    config = tiny(name='ex1')
    streams = RandomStreams(config.seed)
    expected = [_hash_tensors(initialize_model(config.model.widths, init, streams).items())
                for init in INITIALIZATIONS]
    result = run_experiment(config)
    assert [r.metadata['initial_parameter_hash'] for r in result.runs] == expected


@pytest.mark.parametrize('method', ['kfac', 'efb', 'inf'])
def test_custom_jet_adapter_forwarded(method):
    from laplace_hypopinn.curvature.jets import JetLayer, JetTrace
    config = tiny(method)
    parameters = {'coefficient': torch.tensor([[.3, -.7, .2]])}
    def custom(parameters, coordinates):
        return torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), 1) @ parameters['coefficient'].T
    calls = []
    def adapter(parameters, coordinates):
        calls.append(len(coordinates))
        value = torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), 1)
        z = coordinates.new_tensor([1., 0., 0.]).expand_as(value)
        x = coordinates.new_tensor([0., 1., 0.]).expand_as(value)
        inputs = torch.stack((value, z, x), 1)
        pre = tuple(inputs[:, channel] @ parameters['coefficient'].T for channel in range(3))
        return JetTrace((JetLayer(inputs, pre),), torch.stack(pre, 1))
    result = run_experiment(config, parameters=parameters, forward=custom, jet_adapter=adapter)
    assert calls
    assert result.summary.locations.shape == (2, 2)


def test_disk_storage_paths_and_publication_race(tmp_path, monkeypatch):
    import laplace_hypopinn.experiment as experiment
    config = replace(tiny(), save_posterior_fields=True)
    result = run_experiment(config, output=tmp_path/'fields')
    assert result.summary.field_store == str(tmp_path/'fields'/'posterior_fields.npy')
    assert np.load(result.summary.field_store, mmap_mode='r').shape == (2, 4, 5)
    original = experiment._publish
    def collision(staging, target):
        target.mkdir()
        (target/'sentinel').write_text('owned by another run')
        original(staging, target)
    monkeypatch.setattr(experiment, '_publish', collision)
    with pytest.raises(FileExistsError):
        run_experiment(tiny(), output=tmp_path/'collision')
    assert (tmp_path/'collision'/'sentinel').read_text() == 'owned by another run'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['collision', 'fields']


def test_failed_run_does_not_publish_output(tmp_path):
    parameters = {'vector': torch.tensor([.3, .4])}
    with pytest.raises(ValueError):
        run_experiment(tiny('kfac'), parameters=parameters,
                       forward=lambda p, x: x@p['vector'], output=tmp_path/'failed')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['ex3', 'ex3_2', 'ex3_3'])
def test_otway_presets_with_explicit_small_velocity_csv(name, tmp_path):
    pytest.importorskip('skfmm')
    csv = tmp_path/'velocity.csv'
    np.savetxt(csv, np.linspace(2., 3., 35).reshape(5, 7).T, delimiter=',')
    preset = load_preset(name, velocity_path=csv)
    config = tiny()
    preset = replace(preset, model=config.model, training=config.training, laplace=config.laplace,
                     data=replace(preset.data, collocation_count=4, receiver_count=3,
                                  receiver_indices=(0, 3, 6)))
    result = run_experiment(preset)
    assert result.map_field.shape == (5, 7)
    assert result.problem.reference_times is not None
    assert result.metadata['problem_origin'] == 'prepared'


def test_saved_checkpoint_has_explicit_training_only_scope(tmp_path):
    from laplace_hypopinn.artifacts import load_checkpoint
    config = tiny()
    result = run_experiment(config, output=tmp_path/'run')
    state = load_checkpoint(tmp_path/'run'/'checkpoint.pt', config)
    assert state.step == 1
    torch.testing.assert_close(state.parameters, result.parameters)
    assert result.metadata['checkpoint_scope'].startswith('training state only')
    incompatible = replace(config, training=replace(config.training, learning_rate=.001))
    with pytest.raises(ValueError, match='config'):
        load_checkpoint(tmp_path/'run'/'checkpoint.pt', incompatible)


@pytest.mark.parametrize('method,forbidden', [
    ('legacy_diagonal', ['kfac', 'efb', 'inf', 'valla', 'dense', 'jets']),
    ('full', ['kfac', 'efb', 'inf', 'valla', 'diagonal', 'jets']),
    ('efb', ['inf', 'valla', 'dense', 'diagonal']),
    ('valla', ['kfac', 'efb', 'inf', 'dense', 'diagonal', 'jets']),
])
def test_method_dependencies_are_lazy_in_fresh_process(method, forbidden):
    import os
    from pathlib import Path
    import subprocess
    import sys
    program = '''
import sys
from laplace_hypopinn.config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, LaplaceConfig
from laplace_hypopinn.experiment import run_experiment
config = ExperimentConfig(name='lazy-test', model=ModelConfig(widths=(2,2,1)),
    data=DataConfig(collocation_count=2, grid_shape=(3,4), receiver_count=2),
    training=TrainingConfig(steps=1), laplace=LaplaceConfig(method=METHOD, batches=1,
    batch_size=2, samples=1, chunk_size=2, inducing_count=2, variational_steps=1))
run_experiment(config)
for suffix in FORBIDDEN:
    assert 'laplace_hypopinn.curvature.' + suffix not in sys.modules, suffix
'''.replace('METHOD', repr(method)).replace('FORBIDDEN', repr(forbidden))
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src')}
    subprocess.run([sys.executable, '-c', program], env=environment, check=True, capture_output=True, text=True)


def test_broken_symlink_output_is_existing_path(tmp_path):
    destination = tmp_path/'existing-link'
    destination.symlink_to(tmp_path/'absent')
    with pytest.raises(FileExistsError):
        run_experiment(tiny(), output=destination)
    assert not (tmp_path/'absent').exists()
