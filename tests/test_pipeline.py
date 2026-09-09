import csv
import json
import sys
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from transferlab import evaluate, report


class TenClassToy(nn.Module):
    def forward(self, x):
        v = (x.flatten(1).mean(1) - .5) * 10
        return torch.stack([v, -v] + [torch.full_like(v, -100)] * 8, dim=1)


def test_evaluation_and_report_preserve_counts(monkeypatch, tmp_path):
    data = TensorDataset(torch.full((8, 3, 4, 4), .55), torch.zeros(8, dtype=torch.long))
    monkeypatch.setattr(evaluate.datasets, 'CIFAR10', lambda *a, **k: data)
    paths = [tmp_path / 'a.pt', tmp_path / 'b.pt']
    for path in paths:
        path.write_bytes(b'synthetic checkpoint fixture')
    def load(path, device):
        return TenClassToy().eval(), dict(architecture=path.stem, seed=0, epoch=1, validation_accuracy=1.)
    monkeypatch.setattr(evaluate, 'load_checkpoint', load)
    run = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoints', *map(str, paths), '--output', str(run),
                                    '--attacks', 'clean', 'fgsm', '--samples', '8', '--batch-size', '3', '--epsilon', '.2'])
    evaluate.main()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    rows = list(csv.DictReader((run / 'predictions.csv').open()))
    assert len(rows) == 8 * 2 * 2 * 2
    for result in json.loads((run / 'summary.json').read_text()):
        assert result['pair_transfer']['total'] == 8
        assert result['pair_transfer']['rate'] == (0 if result['attack'] == 'clean' else 1)
    example = torch.load(run / 'example.pt', weights_only=True)
    assert example['attack'] == 'fgsm'
    monkeypatch.setattr(sys, 'argv', ['report', str(run)])
    report.main()
    assert '100.00' in (run / 'table.md').read_text()
    table = (run / 'table.md').read_text()
    assert '[8/8]' in table and 'Pilot' in table and 'Max L2' in table
    summaries = json.loads((run / 'summary.json').read_text())
    summaries[0]['pair_transfer']['successes'] = 7
    (run / 'summary.json').write_text(json.dumps(summaries))
    with pytest.raises(ValueError, match='Summary'):
        report.main()
    with (run / 'predictions.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        report.main()


def test_nonfinite_attacked_logits_are_rejected():
    class Unstable(nn.Module):
        def forward(self, x):
            result = torch.zeros(len(x), 10)
            if x.mean() < .5:
                result[:, 0] = float('nan')
            return result
    model = Unstable()
    assert evaluate.predict(model, torch.ones(2, 3, 4, 4)).shape == (2,)
    with pytest.raises(ValueError, match='finite'):
        evaluate.predict(model, torch.zeros(2, 3, 4, 4))


@pytest.mark.parametrize('flag', ['--epsilon', '--step-size', '--l2-budget', '--cw-learning-rate'])
@pytest.mark.parametrize('value', ['nan', 'inf'])
def test_nonfinite_attack_arguments(monkeypatch, tmp_path, flag, value):
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoints', 'unused.pt', '--output', str(tmp_path), flag, value])
    with pytest.raises(SystemExit):
        evaluate.main()
