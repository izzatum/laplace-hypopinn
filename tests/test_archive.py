"""Byte preservation and bounded execution checks for historical notebooks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'archive'
NOTEBOOK_NAMES = {'EX_1.ipynb', 'EX_2.ipynb', 'EX_3.ipynb', 'EX_3_2.ipynb', 'EX_3_3.ipynb', 'EX_3_plot.ipynb'}
RUNNER = ROOT / 'scripts' / 'run_archived_notebook.py'
ORIGINAL_SHA256 = {'EX_1.ipynb': 'bd34c47aa08901f565fdd9d1b8b2691b292bcee2ceefcc46c5b84bf6d68e60f3', 'EX_2.ipynb': 'cee7ccfd0e41dd8d5691666f141b75610a87c65c25aeba2722686fa30e603eea', 'EX_3.ipynb': '1cd0ecc4a065d7d10169c23e287d2f381d9ae6bc8745278168c06ccbc491cee1', 'EX_3_2.ipynb': '8eac8f9e7c1d293cc16fa0fed068aa2623ad009e482ef55a271794ba1e8a8e37', 'EX_3_3.ipynb': 'eff90fa88da5216e9378664fdb74680da9987870f650cb58b80bf3785f2652d6', 'EX_3_plot.ipynb': '32233bb6144157b19e49859b4a5499feea1b94109313919d51c4e76a4e1cecd0', 'curvature/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'curvature/curvatures.py': 'e9a482c633d1529a2f299503d6f2199817edc6e43bca43f0b9b24e48112e74ce', 'curvature/datasets.py': '607fd6cb534189b8fe6ffa33a949ed9abe51d377f407c4fb20b5bbe0014365aa', 'curvature/imagenet.py': '1046cde3d3f72de8658ba33a87d0dc957ae2bd0873d411529fe1be27f466e4c0', 'curvature/lenet5.py': '0d146afc324dd1b7cbb4a28326b6a6063260343bd0856fdeaee47ff3868b4bcb', 'curvature/lenet5_mnist.pth': '221def92f84b4be91f03622b7c88984f6f4b6b8fbfbe35d052675b0a92763b13', 'curvature/resnet.py': 'b6a9a9a4c14075e7c54d43ad1bcaaba8df8c2b776a6a20bfe6cbacb81c8c8199', 'curvature/utils.py': 'fb85a9d56ecd2f6ec47aa3de11742081c1e89627275d71d15625baf140c65c15'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_matches_preserved_files():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    assert manifest['algorithm'] == 'sha256'
    entries = manifest['files']
    assert {e['original_path']: e['sha256'] for e in entries} == ORIGINAL_SHA256
    assert len(entries) == 14
    assert {p.name for p in (ARCHIVE / 'notebooks').glob('*.ipynb')} == NOTEBOOK_NAMES
    actual_paths = {p.relative_to(ROOT).as_posix() for directory in ('notebooks', 'legacy') for p in (ARCHIVE / directory).rglob('*') if p.is_file()}
    assert {e['archived_path'] for e in entries} == actual_paths
    for entry in entries:
        assert digest(ROOT / entry['archived_path']) == entry['sha256']
        assert not (ROOT / entry['original_path']).exists()
    assert (ROOT / 'otway2.csv').is_file()


@pytest.mark.parametrize('name', sorted(NOTEBOOK_NAMES))
def test_selected_original_cells_import_initialize_and_read_data(name, tmp_path):
    notebook = ARCHIVE / 'notebooks' / name
    original = notebook.read_bytes()
    cells = [2, 4, 6] if name == 'EX_3_plot.ipynb' else [2, 4, 6, 10]
    output = tmp_path / name
    run = subprocess.run([sys.executable, str(RUNNER), str(notebook), '--output', str(output), '--cells', ','.join(map(str, cells))], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    executed = json.loads(output.read_text())
    for index, cell in enumerate(executed['cells']):
        if cell['cell_type'] == 'code':
            assert (cell['execution_count'] is not None) == (index in cells)
            assert not any(o.get('output_type') == 'error' for o in cell['outputs'])
    assert notebook.read_bytes() == original


def test_runner_rejects_overwrite_archive_and_source_tree(tmp_path):
    notebook = ARCHIVE / 'notebooks' / 'EX_3.ipynb'
    original = notebook.read_bytes()
    existing = tmp_path / 'existing.ipynb'
    existing.write_text('keep me')
    for output in (notebook, ARCHIVE / 'new.ipynb', ROOT / 'new.ipynb', existing):
        run = subprocess.run([sys.executable, str(RUNNER), str(notebook), '--output', str(output), '--cells', '2'], capture_output=True, text=True, timeout=15)
        assert run.returncode != 0
        assert 'output' in run.stderr.lower()
    assert existing.read_text() == 'keep me'
    assert notebook.read_bytes() == original


@pytest.mark.parametrize('selection', ['-1', '9999', '2,2', 'not-a-cell'])
def test_runner_rejects_invalid_cell_selection(selection, tmp_path):
    output = tmp_path / 'invalid.ipynb'
    run = subprocess.run([sys.executable, str(RUNNER), 'EX_3.ipynb', '--output', str(output), '--cells=' + selection], capture_output=True, text=True, timeout=15)
    assert run.returncode != 0
    assert 'cell' in run.stderr.lower()
    assert not output.exists()
