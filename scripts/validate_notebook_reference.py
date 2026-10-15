"""Run the archived EX2 diagonal reference and compare scientific artifacts.

Install the package and notebook extra first. This is an opt-in research run,
not a claim that long nonconvex float32 optimization is bitwise reproducible.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def compare_array(reference, candidate, *, atol=1e-5, rtol=1e-4):
    """Report actual discrepancies; reject nonfinite or incomplete comparisons."""
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError('comparison tolerances must be finite and nonnegative')
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError('comparison arrays must have identical nonempty shape')
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('comparison arrays must be finite')
    error = candidate - reference
    return {
        'shape': list(reference.shape), 'exact': bool(np.array_equal(reference, candidate)),
        'allclose': bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        'max_absolute_error': float(np.abs(error).max()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'relative_l2_error': float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-300)),
        'atol': atol, 'rtol': rtol,
    }


def load_reference(path, *, steps, samples):
    reference = torch.load(path, map_location='cpu', weights_only=True)
    required = {'initial', 'mean', 'curvature', 'map_field', 'history', 'locations',
                'training_input', 'training_velocity', 'receiver_coordinates', 'receiver_times'}
    if not isinstance(reference, dict) or missing_keys(reference, required):
        raise ValueError('reference is missing required scientific arrays')
    if reference['history'].shape != (steps, 3) or reference['locations'].shape != (samples, 2):
        raise ValueError('reference has incomplete or incompatible training/sample budgets')
    if reference['map_field'].numel() != 101 * 151 or len(reference['curvature']) != 7:
        raise ValueError('reference has an incompatible grid or model')
    return reference


def missing_keys(mapping, required):
    return required - mapping.keys()


def synchronized_time(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def archived_reference(args, directory):
    """Execute unchanged definition/data cells and the original diagonal class."""
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    sys.path.insert(0, str(ROOT / 'archive' / 'legacy'))
    sys.dont_write_bytecode = True
    try:
        from curvature.curvatures import Diagonal
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    namespace = {'torch': torch, 'nn': torch.nn, 'ag': torch.autograd, 'np': np,
                 'device': torch.device(args.device)}
    notebook = ROOT / 'archive' / 'notebooks' / 'EX_2.ipynb'
    cells = json.loads(notebook.read_text())['cells']
    for index in (6, 8, 10, 12):
        exec(compile(''.join(cells[index]['source']), f'{notebook}:cell{index}', 'exec'), namespace)
    model = namespace['PINN']().to(args.device)
    system, batch = namespace['LossEikonal'](), namespace['params']
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.005, betas=(.5, .9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    history = []
    started = synchronized_time(args.device)
    for _ in range(args.steps):
        total, pde, data = system(model, batch)
        total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        history.append([total.item(), pde.item(), data.item()])
    training_seconds = synchronized_time(args.device) - started
    mean = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        field = model(namespace['testset']).detach().cpu().clone()
    diagonal = Diagonal(model)
    started = synchronized_time(args.device)
    for _ in range(args.batches):
        indices = np.random.choice(np.arange(namespace['Z'].numel()), 2501, replace=True)
        coordinates = torch.hstack((namespace['Z'].reshape(-1, 1)[indices],
                                    namespace['X'].reshape(-1, 1)[indices]))
        selected = {'input': coordinates.to(args.device).requires_grad_(True),
                    'vel': namespace['velmodel'].reshape(-1, 1)[indices].to(args.device),
                    'data_pts': namespace['dataset'], 'data': namespace['data']}
        model.zero_grad()
        system(model, selected)[0].backward()
        diagonal.update(2501)
    curvature_seconds = synchronized_time(args.device) - started
    statistic = [value.detach().cpu().clone() for value in diagonal.state.values()]
    diagonal.invert(1000., .001)
    started = synchronized_time(args.device)
    locations = []
    with torch.no_grad():
        for _ in range(args.samples):
            diagonal.sample_and_replace()
            sampled = model(namespace['testset']).reshape(namespace['Z'].shape)
            iz, ix = divmod(int(sampled.argmin()), sampled.shape[1])
            locations.append([.02 * iz, .02 * ix])
    sampling_seconds = synchronized_time(args.device) - started
    result = {'initial': initial, 'mean': mean, 'curvature': statistic, 'map_field': field,
              'history': torch.tensor(history), 'locations': torch.tensor(locations),
              'training_input': batch['input'].detach().cpu(),
              'training_velocity': batch['vel'].cpu(),
              'receiver_coordinates': batch['data_pts'].cpu(), 'receiver_times': batch['data'].cpu()}
    torch.save(result, directory / 'reference.pt')
    metadata = {'training_steps': args.steps, 'curvature_batches': args.batches,
                'posterior_samples': args.samples, 'seed': args.seed, 'device': args.device,
                'threads': args.threads, 'torch': torch.__version__, 'numpy': np.__version__,
                'training_seconds': training_seconds, 'curvature_seconds': curvature_seconds,
                'sampling_seconds': sampling_seconds, 'final_loss': history[-1][0],
                'notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
                'legacy_curvature_sha256': hashlib.sha256(
                    (ROOT / 'archive' / 'legacy' / 'curvature' / 'curvatures.py').read_bytes()
                ).hexdigest()}
    (directory / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return result, metadata


def package_run(args, directory):
    from laplace_hypopinn.experiment import run_experiment
    from laplace_hypopinn.presets import load_preset

    config = load_preset('ex2')
    config = replace(config, seed=args.seed, device=args.device,
                     training=replace(config.training, steps=args.steps, log_every=1),
                     laplace=replace(config.laplace, batches=args.batches, samples=args.samples))
    return run_experiment(config, output=directory / 'package')


def compare_result(reference, result, *, atol, rtol):
    from laplace_hypopinn.data import sample_training_batch
    from laplace_hypopinn.model import flatten_parameters, initialize_model
    from laplace_hypopinn.types import RandomStreams

    def numpy(value):
        return value.detach().cpu().numpy()

    def vector(state):
        remapped = {name.replace('linears.', 'layers.').replace(
            f'layers.{index * 2}.', f'layers.{index}.'): value
                    for index in range(7) for name, value in state.items()
                    if name.startswith(f'linears.{index * 2}.')}
        return flatten_parameters(remapped)[0]

    streams = RandomStreams(result.config.seed, result.config.device)
    initial = initialize_model(result.config.model.widths, result.config.model.initialization,
                               streams, device=result.config.device)
    batch = sample_training_batch(result.problem, result.config.data.collocation_count, streams)
    candidates = {
        'initial_parameters': (vector(reference['initial']), flatten_parameters(initial)[0]),
        'training_coordinates': (reference['training_input'], batch.coordinates),
        'training_velocity': (reference['training_velocity'].reshape(-1), batch.velocity.reshape(-1)),
        'receiver_coordinates': (reference['receiver_coordinates'], result.problem.receiver_coordinates),
        'receiver_times': (reference['receiver_times'].reshape(-1), result.problem.receiver_times.reshape(-1)),
        'trained_parameters': (vector(reference['mean']), flatten_parameters(result.parameters)[0]),
        'legacy_statistic': (torch.cat([value.reshape(-1) for value in reference['curvature']]),
                             result.posterior.metadata['legacy_statistic']),
        'map_field': (reference['map_field'].reshape(-1), result.map_field.reshape(-1)),
        'loss_history': (reference['history'], torch.tensor([
            [entry[key] for key in ('total', 'pde', 'data')] for entry in result.training.history])),
        'posterior_locations': (reference['locations'], result.summary.locations),
        'posterior_location_mean': (reference['locations'].mean(0), result.summary.mean),
        'posterior_location_std': (reference['locations'].std(0, correction=0), result.summary.std),
    }
    return {name: compare_array(numpy(left), numpy(right), atol=atol, rtol=rtol)
            for name, (left, right) in candidates.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new report directory')
    parser.add_argument('--reference', type=Path, help='reuse complete .pt and adjacent .json baseline')
    parser.add_argument('--steps', type=int, default=3001)
    parser.add_argument('--batches', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args(argv)
    if min(args.steps, args.batches, args.samples, args.threads) < 1:
        parser.error('budgets and thread count must be positive')
    torch.set_num_threads(args.threads)
    if args.reference:
        reference = load_reference(args.reference, steps=args.steps, samples=args.samples)
        metadata = json.loads(args.reference.with_suffix('.json').read_text())
        expected = {'training_steps': args.steps, 'curvature_batches': args.batches,
                    'posterior_samples': args.samples, 'seed': args.seed, 'device': args.device,
                    'threads': args.threads}
        for key, value in expected.items():
            # Earlier historical CPU reports omitted these two known defaults.
            actual = metadata.get(key, {'seed': 4321, 'device': 'cpu'}.get(key))
            if actual != value:
                raise ValueError(f'reference {key}={actual!r} differs from requested {value!r}')
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.reference:
        reference, metadata = archived_reference(args, args.output)
    result = package_run(args, args.output)
    comparisons = compare_result(reference, result, atol=args.atol, rtol=args.rtol)
    passed = all(entry['allclose'] for entry in comparisons.values())
    report = {'schema_version': 1, 'command': sys.argv, 'comparison_passed': passed,
              'command_exit_status': 0 if passed else 2, 'reference': metadata,
              'package': result.metadata, 'comparisons': comparisons,
              'note': 'Long float32 Adam trajectories can diverge; inspect actual errors, not just the verdict.'}
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'comparison_passed': passed, 'report': str(args.output / 'comparison.json')}))
    return report['command_exit_status']


if __name__ == '__main__':
    raise SystemExit(main())
