"""Composable research runs with explicit models, random streams and artifacts."""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import torch

from .config import INITIALIZATIONS, resolve_device, resolve_dtype
from .data import prepare_problem, sample_training_batch
from .model import flatten_parameters, initialize_model, make_mlp_forward
from .prediction import grid_location, predict_grid, sample_locations
from .training import train
from .types import RandomStreams


@dataclass(frozen=True)
class ExperimentResult:
    config: object
    problem: object
    training: object
    map_field: torch.Tensor
    map_location: torch.Tensor
    posterior: object | None
    summary: object | None
    metadata: dict = field(default_factory=dict)
    runs: tuple = ()
    output: str | None = None

    @property
    def parameters(self):
        return self.training.parameters


def _hash_tensors(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _problem_hash(problem):
    return _hash_tensors((name, value) for name, value in (
        ('z', problem.grid.z), ('x', problem.grid.x), ('velocity', problem.grid.velocity),
        ('receiver_coordinates', problem.receiver_coordinates), ('receiver_times', problem.receiver_times),
        ('reference_times', problem.reference_times), ('true_source', problem.true_source)) if value is not None)


def _json(value):
    """Summarize tensor diagnostics; never serialize dense curvature into JSON."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'min': float(value.min()) if value.numel() else None,
                'max': float(value.max()) if value.numel() else None,
                'norm': float(value.norm()) if value.numel() else 0.}
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _fit_posterior(parameters, problem, plan, config, streams, forward, jet_adapter):
    settings = config.laplace
    method = settings.method
    if method == 'legacy_diagonal':
        from .curvature.diagonal import fit_legacy_diagonal
        return fit_legacy_diagonal(parameters, problem, settings, streams, forward=forward,
                                   training_config=config.training)
    if method in ('diagonal', 'block_diagonal', 'block', 'full'):
        from .curvature.dense import fit_ggn
        return fit_ggn(parameters, plan, structure=method, prior_precision=settings.prior_precision,
                       scale=settings.scale, max_dense_bytes=settings.max_dense_bytes,
                       chunk_size=settings.chunk_size, forward=forward)
    if method == 'valla':
        from .curvature.valla import fit_valla
        return fit_valla(parameters, plan, settings, streams, forward=forward)
    if method == 'kfac':
        from .curvature.kfac import fit_kfac
        return fit_kfac(parameters, plan, settings, forward=forward, jet_adapter=jet_adapter)
    if method in ('efb', 'inf'):
        from .curvature.kfac import accumulate_factors
        factors = accumulate_factors(parameters, plan, chunk_size=settings.chunk_size,
                                     forward=forward, jet_adapter=jet_adapter)
        if method == 'efb':
            from .curvature.efb import fit_efb
            return fit_efb(parameters, plan, factors, settings, forward=forward, jet_adapter=jet_adapter)
        from .curvature.efb import accumulate_eigenvalues
        from .curvature.inf import fit_inf
        stats = accumulate_eigenvalues(parameters, plan, factors, chunk_size=settings.chunk_size,
                                       forward=forward, jet_adapter=jet_adapter)
        return fit_inf(parameters, factors, stats, settings, forward=forward)
    raise ValueError(f'unsupported Laplace method {method!r}')


def _single_run(config, problem, batch, parameters, streams, forward, jet_adapter,
                uncertainty, field_store):
    started = time.perf_counter()
    initial_hash = _hash_tensors(parameters.items())
    training = train(parameters, batch, config.training, streams, forward=forward)
    training_seconds = time.perf_counter()-started
    field = predict_grid(training.parameters, problem.grid, config.prediction_chunk_size, forward=forward)
    posterior, summary = None, None
    curvature_seconds = prediction_seconds = 0.
    if uncertainty:
        started = time.perf_counter()
        plan = None
        if config.laplace.method != 'legacy_diagonal':
            from .curvature.residuals import make_residual_plan
            plan = make_residual_plan(batch, config.training)
        posterior = _fit_posterior(training.parameters, problem, plan, config, streams, forward, jet_adapter)
        curvature_seconds = time.perf_counter()-started
        started = time.perf_counter()
        summary = sample_locations(posterior, problem.grid, config, streams.torch, field_store=field_store)
        prediction_seconds = time.perf_counter()-started
    metadata = {
        **_provenance(),
        'schema_version': 1, 'method': config.laplace.method if uncertainty else None,
        'initialization': config.model.initialization, 'initial_parameter_hash': initial_hash,
        'collocation_hash': _hash_tensors((('coordinates', batch.coordinates), ('velocity', batch.velocity))),
        'data_hash': _problem_hash(problem), 'parameter_count': flatten_parameters(training.parameters)[1].numel,
        'forward': {'module': getattr(forward, '__module__', None),
                    'name': getattr(forward, '__qualname__', repr(forward)),
                    'model_kind': getattr(forward, 'model_kind', 'custom'),
                    'activation': getattr(forward, 'activation', None)},
        'training_seconds': training_seconds, 'curvature_seconds': curvature_seconds,
        'posterior_prediction_seconds': prediction_seconds,
        'curvature': _json(dict(posterior.metadata)) if posterior is not None else None,
        'posterior_representation': posterior.representation if posterior is not None else None,
        'prediction_kind': posterior.prediction_kind if posterior is not None else None,
        'precision_diagnostics': _precision_diagnostics(posterior) if posterior is not None else None,
        'checkpoint_scope': 'training state only; resume with training API and validate full config/data/forward separately',
        'normalization': 'fixed weighted scalar residual plan' if uncertainty and config.laplace.method != 'legacy_diagonal' else 'legacy batch gradient square' if uncertainty else None,
    }
    if problem.reference_times is not None:
        metadata['map_rmse_seconds'] = float((field-problem.reference_times).square().mean().sqrt())
    location = grid_location(field, problem.grid)
    if problem.true_source is not None:
        metadata['map_location_error_km'] = float((location-problem.true_source).norm())
    return ExperimentResult(config, problem, training, field, location, posterior, summary, metadata)


def _precision_diagnostics(posterior):
    return _json({'diagonal': posterior.diagonal, 'cholesky_factor': posterior.factor,
                  'block_cholesky_factors': posterior.blocks,
                  'kronecker_precision_eigenvalues': tuple(values for _, _, values in posterior.eigenbasis),
                  'low_rank_columns': posterior.low_rank})


def _provenance():
    import subprocess

    try:
        version = importlib.metadata.version('laplace-hypopinn')
    except importlib.metadata.PackageNotFoundError:
        version = 'uninstalled-source'
    revision = None
    source_digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    for source_path in sorted(package_root.rglob('*.py')):
        source_digest.update(str(source_path.relative_to(package_root)).encode())
        source_digest.update(source_path.read_bytes())
    source_root = Path(__file__).resolve().parents[2]
    if (source_root/'.git').exists():
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source_root,
                                      capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {'package_version': version, 'source_revision': revision,
            'source_tree_sha256': source_digest.hexdigest(),
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'python_version': platform.python_version(),
            'created_utc': datetime.now(timezone.utc).isoformat()}


def _save_run(result, directory):
    from .artifacts import save_checkpoint

    directory.mkdir(parents=True, exist_ok=True)
    (directory/'config.json').write_text(json.dumps(asdict(result.config), indent=2)+'\n')
    metrics = {**result.metadata, 'history': result.training.history,
               'map_location': result.map_location.tolist()}
    if result.summary is not None:
        metrics['posterior_location_mean'] = result.summary.mean.tolist()
        metrics['posterior_location_std'] = result.summary.std.tolist()
    (directory/'metrics.json').write_text(json.dumps(_json(metrics), indent=2, allow_nan=False)+'\n')
    arrays = {'z': result.problem.grid.z, 'x': result.problem.grid.x,
              'velocity': result.problem.grid.velocity, 'receiver_coordinates': result.problem.receiver_coordinates,
              'receiver_times': result.problem.receiver_times, 'map_field': result.map_field,
              'map_location': result.map_location}
    for name in ('reference_times', 'true_source'):
        value = getattr(result.problem, name)
        if value is not None:
            arrays[name] = value
    if result.summary is not None:
        arrays.update(posterior_locations=result.summary.locations, posterior_location_mean=result.summary.mean,
                      posterior_location_std=result.summary.std, field_mean=result.summary.field_mean,
                      field_std=result.summary.field_std)
    np.savez_compressed(directory/'arrays.npz', **{key: value.detach().cpu().numpy()
                                                for key, value in arrays.items() if value is not None})
    save_checkpoint(directory/'checkpoint.pt', result.training.state)
    for run in result.runs:
        _save_run(run, directory/run.config.model.initialization)


def _publish(staging, target):
    """Publish a new run without replacing an existing directory.

    Linux supports atomic whole-directory publication; other platforms reserve
    a new directory and publish its already-complete files individually.
    """
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        # Portable fallback: reserve the directory exclusively, then publish
        # complete staged files. Each file is complete when it becomes visible.
        target.mkdir()
        try:
            for source in staging.iterdir():
                source.rename(target/source.name)
            staging.rmdir()
        except BaseException:
            shutil.rmtree(target)
            raise
        return
    if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(target)
        raise OSError(code, os.strerror(code), str(target))


def _relocate_result(result, staging, target):
    summary = result.summary
    if summary is not None and summary.field_store is not None:
        summary = replace(summary, field_store=str(target/Path(summary.field_store).relative_to(staging)))
    return replace(result, summary=summary, runs=tuple(_relocate_result(run, staging, target) for run in result.runs))


def run_experiment(config, *, problem=None, parameters=None, forward=None, jet_adapter=None,
                   output=None, uncertainty=None):
    """Train, fit only the selected estimator, predict and optionally publish a run.

    Supplied models must be deterministic differentiable pointwise scalar maps.
    Structured KFAC/EFB/INF require a supported MLP or an explicit jet adapter.
    EX1 compares four sequential initializations with one fixed collocation set;
    pass uncertainty=True to additionally fit each model's posterior.
    """
    if (parameters is None) != (forward is None):
        raise ValueError('custom parameters and forward must be supplied together')
    if config.name == 'ex1' and parameters is not None:
        raise ValueError('EX1 initialization comparison requires the configured MLP; use another name for a custom model')
    requested = Path(output).expanduser() if output is not None else None
    if requested is not None and (requested.exists() or requested.is_symlink()):
        raise FileExistsError(requested)
    target = requested.resolve() if requested is not None else None
    if config.save_posterior_fields and target is None:
        raise ValueError('save_posterior_fields requires an output directory')
    staging = None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.'+target.name+'.', dir=target.parent))
    try:
        dtype, device = resolve_dtype(config.dtype), resolve_device(config.device)
        prepared_problem = problem is None
        problem = prepare_problem(config) if prepared_problem else problem
        tensors = [problem.grid.z, problem.grid.x, problem.grid.velocity,
                   problem.receiver_coordinates, problem.receiver_times]
        if any(value.dtype != dtype or value.device != device for value in tensors):
            raise ValueError('supplied problem dtype/device must match experiment configuration')
        streams = RandomStreams(config.seed, device)
        # Notebook EX1 consumes the first initialization before selecting the one
        # fixed collocation set; later initializations share the same torch stream.
        first_initialization = INITIALIZATIONS[0] if config.name == 'ex1' else config.model.initialization
        parameters = initialize_model(config.model.widths, first_initialization, streams,
                                      dtype=dtype, device=device) if parameters is None else parameters
        if any(value.dtype != dtype or value.device != device for value in parameters.values()):
            raise ValueError('supplied parameters dtype/device must match experiment configuration')
        forward = make_mlp_forward(config.model.activation) if forward is None else forward
        batch = sample_training_batch(problem, config.data.collocation_count, streams)
        uncertainty = config.name != 'ex1' if uncertainty is None else uncertainty
        runs = []
        if config.name == 'ex1':
            for index, initialization in enumerate(INITIALIZATIONS):
                current = replace(config, model=replace(config.model, initialization=initialization))
                if index == 0:
                    current_parameters = parameters
                else:
                    current_parameters = initialize_model(current.model.widths, initialization, streams,
                                                          dtype=dtype, device=device)
                run_dir = staging/initialization if staging is not None else None
                if run_dir is not None:
                    run_dir.mkdir()
                runs.append(_single_run(current, problem, batch, current_parameters, streams, forward,
                                        jet_adapter, uncertainty,
                                        run_dir/'posterior_fields.npy' if config.save_posterior_fields else None))
            result = replace(runs[0], config=config, runs=tuple(runs))
        else:
            result = _single_run(config, problem, batch, parameters, streams, forward, jet_adapter,
                                 uncertainty, staging/'posterior_fields.npy' if config.save_posterior_fields else None)
        result = replace(result, metadata={**result.metadata,
                         'problem_origin': 'prepared' if prepared_problem else 'supplied'},
                         output=str(target) if target is not None else None)
        if staging is not None:
            _save_run(result, staging)
            _publish(staging, target)
            result = _relocate_result(result, staging, target)
            staging = None
        return result
    finally:
        if staging is not None:
            shutil.rmtree(staging)
