import json

from laplace_hypopinn.cli import main


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
