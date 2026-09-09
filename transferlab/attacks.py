import torch
import torch.nn.functional as F


def linf_attack(model, x, y, epsilon, steps=1, step_size=None, restarts=1, random_start=False):
    if not 0 <= epsilon <= 1 or steps < 1 or restarts < 1:
        raise ValueError('Invalid attack budget, steps, or restarts')
    alpha = epsilon if step_size is None else step_size
    if alpha < 0:
        raise ValueError('Step size must be nonnegative')
    original = x.detach()
    best = original.clone()
    with torch.no_grad():
        best_loss = F.cross_entropy(model(best), y, reduction='none')
        best_success = model(best).argmax(1).ne(y)
    for _ in range(restarts):
        adv = original.clone()
        if random_start:
            adv = (adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)).clamp(0, 1)
        for iteration in range(steps + 1):
            adv = adv.detach().requires_grad_(iteration < steps)
            logits = model(adv)
            loss = F.cross_entropy(logits, y, reduction='none')
            with torch.no_grad():
                success = logits.argmax(1).ne(y)
                better = (success & ~best_success) | ((success == best_success) & (loss > best_loss))
                best[better] = adv.detach()[better]
                best_loss[better] = loss.detach()[better]
                best_success[better] = success[better]
            if iteration < steps:
                grad, = torch.autograd.grad(loss.sum(), adv)
                adv = adv.detach() + alpha * grad.sign()
                adv = torch.maximum(torch.minimum(adv, original + epsilon), original - epsilon).clamp(0, 1)
    return best.detach()


def generate(model, x, y, name, args):
    if name == 'clean':
        return x.detach().clone()
    if name == 'noise':
        return (x + torch.empty_like(x).uniform_(-args.epsilon, args.epsilon)).clamp(0, 1)
    if name == 'fgsm':
        # FGSM is exactly one gradient step, without best-iterate selection.
        z = x.detach().clone().requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(z), y), z)
        return (z.detach() + args.epsilon * grad.sign()).clamp(0, 1)
    if name == 'pgd':
        return linf_attack(model, x, y, args.epsilon, args.steps, args.step_size, args.restarts, True)
    if name == 'cw':
        import foolbox as fb
        attack = fb.attacks.L2CarliniWagnerAttack(steps=args.cw_steps,
                    binary_search_steps=args.cw_search, stepsize=args.cw_learning_rate,
                    confidence=0, abort_early=False)
        _, clipped, _ = attack(fb.PyTorchModel(model, bounds=(0, 1)), x, y, epsilons=args.l2_budget)
        return clipped.detach()
    raise ValueError(name)


def validate_perturbations(x, adv, name, epsilon, l2_budget):
    if not torch.isfinite(adv).all() or adv.min() < -1e-6 or adv.max() > 1 + 1e-6:
        raise ValueError('Attack produced nonfinite or out-of-bounds pixels')
    delta = (adv - x).flatten(1)
    linf, l2 = delta.abs().amax(1), delta.norm(p=2, dim=1)
    budget = l2_budget if name == 'cw' else (0 if name == 'clean' else epsilon)
    norm = l2 if name == 'cw' else linf
    if (norm > budget + 1e-5).any():
        raise ValueError('Attack violated its declared norm budget')
    return linf, l2
