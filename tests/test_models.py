import pytest
import torch
from transferlab.models import CIFARModel, NAMES, load_checkpoint


@pytest.mark.parametrize('name', NAMES)
def test_native_cifar_logits_and_input_gradient(name):
    model = CIFARModel(name).eval()
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    logits = model(x)
    assert logits.shape == (1, 10)
    gradient, = torch.autograd.grad(logits.sum(), x)
    assert torch.isfinite(gradient).all()


def test_rejects_unverified_checkpoint(tmp_path):
    path = tmp_path / 'wrong.pt'
    torch.save({'state_dict': {}}, path)
    with pytest.raises(ValueError, match='CIFAR-10'):
        load_checkpoint(path, 'cpu')


def test_checkpoint_roundtrip(tmp_path):
    from transferlab.common import CLASSES, environment
    from transferlab.models import SPEC
    original = CIFARModel('resnet18').eval()
    path = tmp_path / 'valid.pt'
    torch.save(dict(spec=SPEC, classes=CLASSES, architecture='resnet18',
                    state_dict=original.state_dict(), environment=environment()), path)
    restored, _ = load_checkpoint(path, 'cpu')
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(original(x), restored(x))
