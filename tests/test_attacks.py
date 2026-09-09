from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transferlab.attacks import generate, linf_attack, validate_perturbations


class Toy(nn.Module):
    def forward(self, x):
        v = x.flatten(1).mean(1) - .5
        return torch.stack([v, -v], 1) * 10


def args():
    return SimpleNamespace(epsilon=.2, steps=5, step_size=.05, restarts=2,
                           l2_budget=1., cw_steps=20, cw_search=2, cw_learning_rate=.01)


@pytest.mark.parametrize('attack', ['clean', 'noise', 'fgsm', 'pgd', 'cw'])
def test_attack_bounds_and_finiteness(attack):
    torch.manual_seed(4)
    x = torch.full((2, 3, 4, 4), .55)
    y = torch.zeros(2, dtype=torch.long)
    adv = generate(Toy().eval(), x, y, attack, args())
    validate_perturbations(x, adv, attack, .2, 1.)
    assert not adv.requires_grad
    assert torch.equal(x, torch.full_like(x, .55))


@pytest.mark.parametrize('attack', ['clean', 'noise', 'fgsm', 'pgd'])
def test_zero_budget_identity(attack):
    a = args()
    a.epsilon = 0
    x = torch.rand(2, 3, 4, 4)
    assert torch.equal(generate(Toy().eval(), x, torch.tensor([0, 1]), attack, a), x)


def test_fgsm_direction_and_pgd_success():
    x = torch.full((2, 3, 4, 4), .55)
    y = torch.zeros(2, dtype=torch.long)
    for attack in ['fgsm', 'pgd']:
        adv = generate(Toy().eval(), x, y, attack, args())
        assert Toy()(adv).argmax(1).eq(1).all()
    with pytest.raises(ValueError):
        linf_attack(Toy(), x, y, -1)
    with pytest.raises(ValueError):
        validate_perturbations(x, x + .3, 'fgsm', .2, 1.)


@pytest.mark.parametrize('attack', ['fgsm', 'pgd'])
def test_nonfinite_source_gradient_fails_closed(attack):
    class Singular(nn.Module):
        def forward(self, x):
            v = ((x.flatten(1).mean(1) - .5) ** 2).sqrt()
            return torch.stack([v, -v], dim=1)
    a = args()
    a.epsilon = 0
    with pytest.raises(ValueError, match='gradient'):
        generate(Singular(), torch.full((2, 3, 4, 4), .5), torch.zeros(2, dtype=torch.long), attack, a)


def test_pgd_rejects_nonfinite_intermediate_iterate():
    class Unstable(Toy):
        def forward(self, x):
            logits = super().forward(x)
            return logits * (float('nan') if x.mean() < .5 else 1)
    with pytest.raises(ValueError, match='logits'):
        linf_attack(Unstable(), torch.full((2, 3, 4, 4), .55), torch.zeros(2, dtype=torch.long), .2)
