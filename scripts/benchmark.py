"""Measure selected curvature fits and streamed predictions in isolated workers.

The default uses the notebook's 2,737-parameter model and full EX2 prediction
grid, with short fitting budgets. It does not measure posterior calibration.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

METHODS = ('legacy_diagonal', 'diagonal', 'block_diagonal', 'full', 'kfac', 'efb', 'inf', 'valla')


def fit_method(parameters, problem, plan, config, streams):
    from laplace_hypopinn.curvature.dense import fit_ggn
    from laplace_hypopinn.curvature.diagonal import fit_legacy_diagonal
    from laplace_hypopinn.curvature.efb import accumulate_eigenvalues, fit_efb
    from laplace_hypopinn.curvature.inf import fit_inf
    from laplace_hypopinn.curvature.kfac import accumulate_factors, fit_kfac
    from laplace_hypopinn.curvature.valla import fit_valla

    settings = config.laplace
    method = settings.method
    if method == 'legacy_diagonal':
        return fit_legacy_diagonal(parameters, problem, settings, streams,
                                   training_config=config.training)
    if method in ('diagonal', 'block_diagonal', 'full'):
        return fit_ggn(parameters, plan, structure=method,
                       prior_precision=settings.prior_precision, scale=settings.scale,
                       chunk_size=settings.chunk_size, max_dense_bytes=settings.max_dense_bytes)
    if method == 'kfac':
        return fit_kfac(parameters, plan, settings)
    if method == 'valla':
        return fit_valla(parameters, plan, settings, streams)
    basis = accumulate_factors(parameters, plan, chunk_size=settings.chunk_size)
    if method == 'efb':
        return fit_efb(parameters, plan, basis, settings)
    stats = accumulate_eigenvalues(parameters, plan, basis, chunk_size=settings.chunk_size)
    return fit_inf(parameters, basis, stats, settings)


def summarize(values):
    return {'median': statistics.median(values), 'minimum': min(values), 'runs': values}


def cpu_model():
    info = Path('/proc/cpuinfo')
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith('model name'):
                return line.partition(':')[2].strip()
    return platform.processor() or platform.machine()


def worker(args):
    import numpy as np
    import torch

    from laplace_hypopinn.curvature.residuals import make_residual_plan
    from laplace_hypopinn.data import prepare_problem, sample_training_batch
    from laplace_hypopinn.model import flatten_parameters, initialize_model
    from laplace_hypopinn.prediction import sample_locations
    from laplace_hypopinn.presets import load_preset
    from laplace_hypopinn.training import train
    from laplace_hypopinn.types import RandomStreams

    torch.set_num_threads(args.threads)
    config = load_preset(args.preset, velocity_path=args.velocity_path)
    config = replace(config, seed=args.seed, device=args.device,
                     data=replace(config.data, collocation_count=args.collocation),
                     training=replace(config.training, steps=args.steps),
                     laplace=replace(config.laplace, method=args.worker, batch_size=args.collocation,
                                     batches=args.batches, samples=args.samples,
                                     chunk_size=args.chunk_size, rank=args.rank,
                                     inducing_count=args.inducing_count,
                                     variational_steps=args.variational_steps),
                     prediction_chunk_size=args.prediction_chunk_size)
    problem = prepare_problem(config)
    streams = RandomStreams(args.seed, args.device)
    parameters = initialize_model(config.model.widths, config.model.initialization,
                                  streams, device=args.device)
    batch = sample_training_batch(problem, args.collocation, streams)

    def now():
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)
        return time.perf_counter()

    started = now()
    parameters = train(parameters, batch, config.training, streams).parameters
    training_seconds = now() - started
    plan = make_residual_plan(batch, config.training)
    times = {'fit_seconds': [], 'prediction_seconds': []}
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    for iteration in range(args.warmup + args.repeats):
        # Every repeat uses identical fitting rows and sample randomness.
        local = RandomStreams(args.seed + 1, args.device)
        started = now()
        posterior = fit_method(parameters, problem, plan, config, local)
        fitted = now()
        summary = sample_locations(posterior, problem.grid, config, local.torch)
        predicted = now()
        if not torch.isfinite(summary.locations).all() or not torch.isfinite(summary.field_std).all():
            raise ValueError('benchmark produced nonfinite predictions')
        if iteration >= args.warmup:
            times['fit_seconds'].append(fitted - started)
            times['prediction_seconds'].append(predicted - fitted)
        # Do not hold two posterior structures during the next fit.
        del posterior, summary
    package = Path(importlib.import_module('laplace_hypopinn').__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'schema_version': 1, 'method': args.worker, 'command_exit_status': 0,
        'config': asdict(config), 'parameter_count': flatten_parameters(parameters)[1].numel,
        'grid_points': problem.grid.velocity.numel(), 'warmup': args.warmup, 'repeats': args.repeats,
        'training_seconds': training_seconds, **{key: summarize(value) for key, value in times.items()},
        'process_peak_rss_bytes': int(maximum_rss if sys.platform == 'darwin' else maximum_rss * 1024),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)
        if torch.device(args.device).type == 'cuda' else None,
        'memory_scope': 'worker process high-water RSS, including imports, training, warmup and repeats',
        'finite_predictions': True, 'source_tree_sha256': digest.hexdigest(),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'package': importlib.metadata.version('laplace-hypopinn')},
        'platform': platform.platform(), 'cpu': cpu_model(), 'threads': args.threads,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new benchmark directory')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--preset', choices=('ex2', 'ex3'), default='ex2')
    parser.add_argument('--velocity-path')
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--collocation', type=int, default=32)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=16)
    parser.add_argument('--prediction-chunk-size', type=int, default=4096)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--inducing-count', type=int, default=4)
    parser.add_argument('--variational-steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    positive = ('threads', 'steps', 'collocation', 'batches', 'samples', 'chunk_size',
                'prediction_chunk_size', 'inducing_count', 'variational_steps', 'repeats')
    if any(getattr(args, name) < 1 for name in positive) or min(args.rank, args.warmup) < 0:
        parser.error('budgets must be positive; rank and warmup may be zero')
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    arguments = sys.argv[1:] if argv is None else argv
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    for method in args.methods:
        command = [sys.executable, str(Path(__file__).resolve()), *arguments, '--worker', method]
        process = subprocess.run(command, env=environment, capture_output=True, text=True)
        if process.returncode:
            record = {'method': method, 'command_exit_status': process.returncode,
                      'stdout': process.stdout, 'stderr': process.stderr}
        else:
            record = json.loads(process.stdout)
        results.append(record)
        (args.output / f'{method}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'method': method, 'command_exit_status': process.returncode}), flush=True)
    status = int(any(result['command_exit_status'] for result in results))
    report = {'schema_version': 1, 'command': [sys.executable, *sys.argv],
              'command_exit_status': status, 'results': results,
              'scope': 'short fixed-budget throughput; no accuracy, calibration or cross-method speed equivalence claim'}
    (args.output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
