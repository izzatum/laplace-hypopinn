import json
from types import SimpleNamespace

import torch

from laplace_hypopinn.cli import main


def test_velocity_path_overrides_loaded_config(tmp_path, monkeypatch):
    configuration = tmp_path / 'experiment.toml'
    configuration.write_text('[data]\nkind="otway"\nvelocity_path="original.csv"\n')
    monkeypatch.chdir(tmp_path)
    seen = []

    def run(config, **kwargs):
        seen.append(config)
        return SimpleNamespace(output=kwargs['output'], map_location=torch.zeros(2),
                               metadata={'method': config.laplace.method})

    monkeypatch.setattr('laplace_hypopinn.experiment.run_experiment', run)
    assert main(['run', '--config', str(configuration), '--velocity-path', 'replacement.csv',
                 '--output', str(tmp_path / 'run')]) == 0
    assert seen[0].data.velocity_path == str(tmp_path / 'replacement.csv')


def test_presets(capsys):
    assert main(['presets']) == 0
    assert 'ex3_3' in capsys.readouterr().out


def test_cli_smoke(tmp_path):
    destination = tmp_path/'run'
    assert main(['run', '--preset', 'ex2', '--output', str(destination), '--smoke',
                 '--steps', '1', '--samples', '2', '--method', 'diagonal', '--threads', '1',
                 '--widths', '2,3,1']) == 0
    config = json.loads((destination/'config.json').read_text())
    assert config['model']['widths'] == [2, 3, 1]
    assert config['training']['steps'] == 1
