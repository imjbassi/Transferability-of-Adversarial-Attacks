import json
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from transferlab import train


def test_pilot_training_preserves_split_and_marks_artifacts(monkeypatch, tmp_path):
    data = TensorDataset(torch.rand(50000, 3, 2, 2), torch.zeros(50000, dtype=torch.long))
    monkeypatch.setattr(train.datasets, 'CIFAR10', lambda *a, **k: data)
    monkeypatch.setattr(train, 'CIFARModel', lambda name: nn.Sequential(nn.Flatten(), nn.Linear(12, 10)))
    output = tmp_path / 'pilot'
    monkeypatch.setattr(sys, 'argv', ['train', '--architecture', 'resnet18', '--output', str(output),
                                    '--epochs', '1', '--train-samples', '8', '--validation-samples', '4', '--device', 'cpu'])
    train.main()
    metadata = json.loads((output / 'metadata.json').read_text())
    permutation = torch.randperm(50000, generator=torch.Generator().manual_seed(1729)).tolist()
    assert metadata['train_indices'] == permutation[5000:5008]
    assert metadata['validation_indices'] == permutation[:4]
    assert metadata['pilot'] and metadata['status'] == 'complete'
    checkpoint = torch.load(output / 'best.pt', weights_only=True)
    assert checkpoint['pilot'] and checkpoint['epoch'] == 1
    assert checkpoint['validation_indices'] == metadata['validation_indices']


def test_mobile_singleton_training_batch_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['train', '--architecture', 'mobilenet_v2', '--output', str(tmp_path / 'bad'),
                                    '--train-samples', '129', '--batch-size', '128'])
    with pytest.raises(SystemExit):
        train.main()
    assert not (tmp_path / 'bad').exists()
