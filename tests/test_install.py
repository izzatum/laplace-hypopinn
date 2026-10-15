"""Distribution and validation-tool contracts, exercised away from the checkout."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def validation_module():
    spec = importlib.util.spec_from_file_location(
        'reference_validation', ROOT / 'scripts' / 'validate_notebook_reference.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_comparison_reports_error_and_rejects_incomplete_arrays():
    module = validation_module()
    result = module.compare_array(np.array([1., 2.]), np.array([1., 2.01]), atol=.001, rtol=0.)
    assert result['shape'] == [2]
    assert result['max_absolute_error'] == pytest.approx(.01)
    assert not result['allclose']
    assert not result['exact']
    assert module.compare_array(np.zeros(2), np.zeros(2))['exact']
    for wrong in (np.ones(3), np.array([np.nan, 1.]), np.array([np.inf, 1.])):
        with pytest.raises(ValueError, match='(shape|finite)'):
            module.compare_array(np.ones(2), wrong)
    with pytest.raises(ValueError, match='nonnegative'):
        module.compare_array(np.ones(2), np.ones(2), atol=-1)


def test_reference_loader_rejects_incomplete_run(tmp_path):
    import torch

    module = validation_module()
    path = tmp_path / 'reference.pt'
    torch.save({'history': torch.zeros((2, 3))}, path)
    with pytest.raises(ValueError, match='missing'):
        module.load_reference(path, steps=2, samples=3)


def test_reference_command_writes_complete_schema_and_rejects_wrong_budget(tmp_path):
    output = tmp_path / 'validation'
    command = [sys.executable, str(ROOT / 'scripts' / 'validate_notebook_reference.py'),
               '--output', str(output), '--steps', '2', '--batches', '1', '--samples', '2']
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / 'comparison.json').read_text())
    assert report['schema_version'] == 1
    assert report['command_exit_status'] == 0
    assert report['comparison_passed']
    assert report['package']['parameter_count'] == 2737
    assert report['comparisons']['initial_parameters']['exact']
    assert report['comparisons']['training_coordinates']['exact']
    wrong = [sys.executable, str(ROOT / 'scripts' / 'validate_notebook_reference.py'),
             '--output', str(tmp_path / 'wrong'), '--reference', str(output / 'reference.pt'),
             '--steps', '3', '--batches', '1', '--samples', '2']
    process = subprocess.run(wrong, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert 'incomplete or incompatible' in process.stderr
    assert not (tmp_path / 'wrong').exists()


def test_benchmark_report_records_budget_and_real_worker_status(tmp_path):
    output = tmp_path / 'benchmark'
    command = [sys.executable, str(ROOT / 'scripts' / 'benchmark.py'), '--output', str(output),
               '--methods', 'diagonal', '--steps', '1', '--collocation', '4', '--samples', '1',
               '--warmup', '0', '--repeats', '1']
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / 'benchmark.json').read_text())
    assert report['schema_version'] == 1
    assert report['command_exit_status'] == 0
    result, = report['results']
    assert result['parameter_count'] == 2737
    assert result['grid_points'] == 101 * 151
    assert result['finite_predictions']
    assert result['process_peak_rss_bytes'] > 0
    assert result['fit_seconds']['median'] > 0
    assert result['prediction_seconds']['median'] > 0


def test_benchmark_worker_failure_is_recorded_and_propagated(tmp_path):
    output = tmp_path / 'failed-benchmark'
    process = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'benchmark.py'), '--output', str(output),
         '--methods', 'diagonal', '--preset', 'ex3',
         '--velocity-path', str(tmp_path / 'missing.csv')],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode != 0
    report = json.loads((output / 'benchmark.json').read_text())
    assert report['command_exit_status'] != 0
    result, = report['results']
    assert result['command_exit_status'] != 0
    assert 'missing.csv' in result['stderr']


@pytest.fixture(scope='module')
def installed_wheel(tmp_path_factory):
    directory = tmp_path_factory.mktemp('distribution')
    wheels = directory / 'wheels'
    subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                    '--outdir', str(wheels)], cwd=ROOT, check=True, capture_output=True)
    wheel, = wheels.glob('*.whl')
    target = directory / 'site'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(target),
                    str(wheel)], cwd=directory, check=True, capture_output=True)
    environment = dict(os.environ, PYTHONPATH=str(target), PYTHONDONTWRITEBYTECODE='1')
    return directory, target, environment


def test_installed_wheel_console_runs_outside_checkout(installed_wheel):
    directory, target, environment = installed_wheel
    command = target / 'bin' / 'hypopinn'
    probe = subprocess.run([sys.executable, '-c',
                            'import laplace_hypopinn; print(laplace_hypopinn.__file__)'],
                           cwd=directory, env=environment, check=True, capture_output=True, text=True)
    assert Path(probe.stdout.strip()).is_relative_to(target)
    result = subprocess.run([str(command), 'run', '--preset', 'ex2', '--smoke', '--steps', '1',
                             '--samples', '2', '--threads', '1',
                             '--widths', '2,16,16,32,32,16,16,1', '--output', str(directory / 'run')],
                            cwd=directory, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads((directory / 'run' / 'metrics.json').read_text())
    assert metrics['parameter_count'] == 2737
    assert metrics['method'] == 'legacy_diagonal'


def test_installed_wheel_optional_fast_marching_is_lazy_and_actionable(installed_wheel):
    directory, _, environment = installed_wheel
    # Deny the optional module even when the test host installed the extra.
    script = '''
import importlib.abc
import sys
class MissingFMM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'skfmm':
            raise ModuleNotFoundError('test blocks optional scikit-fmm')
sys.meta_path.insert(0, MissingFMM())
from laplace_hypopinn.data import prepare_problem
from laplace_hypopinn.presets import load_preset
assert 'skfmm' not in sys.modules
prepare_problem(load_preset('ex2'))
try:
    prepare_problem(load_preset('ex3', velocity_path=sys.argv[1]))
except ImportError as error:
    assert 'fast-marching' in str(error), str(error)
else:
    raise AssertionError('Otway must explain its missing dependency')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'otway2.csv')],
                            cwd=directory, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_wheel_accepts_explicit_otway_data_path(installed_wheel):
    pytest.importorskip('skfmm')
    directory, target, environment = installed_wheel
    output = directory / 'otway'
    result = subprocess.run(
        [str(target / 'bin' / 'hypopinn'), 'run', '--preset', 'ex3', '--steps', '1',
         '--batches', '1', '--samples', '1', '--threads', '1',
         '--velocity-path', str(ROOT / 'otway2.csv'), '--output', str(output)],
        cwd=directory, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(output / 'arrays.npz') as arrays:
        assert arrays['velocity'].shape == (246, 305)
        assert np.isfinite(arrays['map_field']).all()
