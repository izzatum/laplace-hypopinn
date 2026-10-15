#!/usr/bin/env python3
"""Execute historical notebooks without changing their archived input bytes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPOSITORY / 'archive' / 'notebooks'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', help='Archived notebook basename or path')
    parser.add_argument('--output', type=Path, required=True, help='New executed copy outside the repository')
    parser.add_argument('--cells', help='Comma-separated original zero-based cell indices; default: all cells')
    parser.add_argument('--kernel', default='laplace-hypopinn', help='Jupyter kernel name (default: laplace-hypopinn)')
    parser.add_argument('--timeout', type=int, default=600, help='Maximum seconds per cell (default: 600)')
    args = parser.parse_args()
    source = Path(args.notebook)
    if source.name == args.notebook:
        source = NOTEBOOKS / source
    source = source.resolve()
    if source.parent != NOTEBOOKS or not source.is_file():
        parser.error('notebook must be a file in archive/notebooks')
    output = args.output.resolve()
    if output.is_relative_to(REPOSITORY):
        parser.error('output must be outside the repository, including its archive')
    if output.exists():
        parser.error('output already exists; choose a new path')
    if args.timeout <= 0:
        parser.error('timeout must be positive')

    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(source, as_version=4)
    if args.cells is None:
        indices = list(range(len(notebook.cells)))
    else:
        try:
            indices = [int(index) for index in args.cells.split(',')]
        except ValueError:
            parser.error('cells must be comma-separated integer indices')
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(notebook.cells) for i in indices):
            parser.error('cell indices must be unique and within the original notebook bounds')
    # Discard saved results in the executed copy so unselected cells cannot look run.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    environment = os.environ.copy()
    legacy = str(REPOSITORY / 'archive' / 'legacy')
    environment['PYTHONPATH'] = legacy + os.pathsep + environment.get('PYTHONPATH', '')
    environment.setdefault('MPLBACKEND', 'Agg')
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    client = NotebookClient(notebook, kernel_name=args.kernel, timeout=args.timeout)
    # Keep original cell indices and source text, with one shared kernel session.
    with client.setup_kernel(cwd=str(REPOSITORY), env=environment):
        for index in indices:
            client.execute_cell(notebook.cells[index], index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as destination:
        nbformat.write(notebook, destination)
    print(output)


if __name__ == '__main__':
    main()
