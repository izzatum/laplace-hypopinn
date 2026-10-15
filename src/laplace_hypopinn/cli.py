"""Command line entry points; importing the package runs no experiments."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import LaplaceConfig
from .presets import load_config, load_preset

PRESETS = ('ex1', 'ex2', 'ex3', 'ex3_2', 'ex3_3')
METHODS = ('legacy_diagonal', 'diagonal', 'block_diagonal', 'full', 'kfac', 'efb', 'inf', 'valla')


def _parser():
    parser = argparse.ArgumentParser(prog='hypopinn')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('presets', help='list historical research configurations')
    run = sub.add_parser('run', help='run an explicit full configuration or short smoke experiment')
    source = run.add_mutually_exclusive_group()
    source.add_argument('--config', help='TOML experiment configuration')
    source.add_argument('--preset', choices=PRESETS)
    run.add_argument('--output', required=True, help='new artifact directory; existing paths are rejected')
    run.add_argument('--velocity-path', help='explicit Otway CSV path')
    run.add_argument('--smoke', action='store_true', help='use a tiny analytic grid and short budget')
    run.add_argument('--steps', type=int)
    run.add_argument('--batches', type=int)
    run.add_argument('--samples', type=int)
    run.add_argument('--method', choices=METHODS)
    run.add_argument('--device')
    run.add_argument('--threads', type=int)
    run.add_argument('--widths', help='comma-separated layer widths, e.g. 2,16,16,1')
    run.add_argument('--uncertainty', action='store_true', default=None,
                     help='also fit uncertainty for EX1 initialization comparisons')
    run.add_argument('--save-posterior-fields', action='store_true',
                     help='stream complete sample fields to posterior_fields.npy on disk')
    plot = sub.add_parser('plot', help='plot an existing run with the optional plotting extra')
    plot.add_argument('run_dir')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == 'presets':
        print('\n'.join(PRESETS))
        return 0
    if args.command == 'plot':
        from .plotting import plot_run
        plot_run(args.run_dir)
        return 0
    if args.threads is not None:
        if args.threads < 1:
            parser.error('--threads must be positive')
        import torch
        torch.set_num_threads(args.threads)
    config = load_config(args.config) if args.config else load_preset(args.preset or 'ex2', velocity_path=args.velocity_path)
    if args.velocity_path is not None:
        config = replace(config, data=replace(config.data,
                         velocity_path=str(Path(args.velocity_path).expanduser().resolve())))
    # With no explicit research configuration, the command defaults to a quick
    # executable example. An explicit preset/config retains full research budgets.
    if args.smoke or (args.config is None and args.preset is None):
        config = replace(config, model=replace(config.model, widths=(2, 4, 1)),
                         data=replace(config.data, kind='gradient', velocity_path=None,
                                      collocation_count=8, grid_shape=(9, 11), receiver_count=5,
                                      receiver_indices=None, z_bounds=(0., 2.), x_bounds=(0., 3.)),
                         training=replace(config.training, steps=2, log_every=1),
                         laplace=LaplaceConfig(method=config.laplace.method, batches=1,
                                              batch_size=8, samples=3, chunk_size=4,
                                              rank=2, inducing_count=2, variational_steps=2))
    training = replace(config.training, **({'steps': args.steps} if args.steps is not None else {}))
    laplace_overrides = {key: getattr(args, key) for key in ('batches', 'samples', 'method')
                         if getattr(args, key) is not None}
    model = config.model
    if args.widths is not None:
        try:
            model = replace(model, widths=tuple(int(value) for value in args.widths.split(',')))
        except ValueError as exc:
            parser.error(str(exc))
    config = replace(config, model=model, training=training,
                     laplace=replace(config.laplace, **laplace_overrides),
                     device=args.device or config.device,
                     save_posterior_fields=args.save_posterior_fields or config.save_posterior_fields)
    from .experiment import run_experiment
    result = run_experiment(config, output=args.output, uncertainty=args.uncertainty)
    print(json.dumps({'output': result.output, 'map_location': result.map_location.tolist(),
                      'method': result.metadata['method']}))
    return 0
