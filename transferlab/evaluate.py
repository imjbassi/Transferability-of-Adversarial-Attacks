import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .attacks import generate, validate_perturbations
from .common import digest, environment, seed_all, write_json
from .metrics import summarize
from .models import load_checkpoint


def main():
    p = argparse.ArgumentParser(description='Evaluate frozen CIFAR-10 checkpoints on identical test samples')
    p.add_argument('--checkpoints', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--attacks', nargs='+', choices=['clean', 'noise', 'fgsm', 'pgd', 'cw'], default=['clean', 'noise', 'fgsm', 'pgd'])
    p.add_argument('--samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epsilon', type=float, default=8/255)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--step-size', type=float, default=2/255)
    p.add_argument('--restarts', type=int, default=5)
    p.add_argument('--l2-budget', type=float, default=1.0)
    p.add_argument('--cw-steps', type=int, default=1000)
    p.add_argument('--cw-search', type=int, default=9)
    p.add_argument('--cw-learning-rate', type=float, default=0.01)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if not 1 <= a.samples <= 10000 or a.batch_size < 1 or not 0 <= a.epsilon <= 1:
        p.error('Invalid sample count, batch size, or epsilon')
    if min(a.steps, a.restarts, a.cw_steps, a.cw_search) < 1 or min(a.l2_budget, a.step_size) < 0 or a.cw_learning_rate <= 0:
        p.error('Invalid attack parameters')
    if len(set(a.attacks)) != len(a.attacks):
        p.error('Duplicate attacks are not allowed')
    seed_all(a.seed)
    models, checkpoints = {}, {}
    for path in a.checkpoints:
        model, info = load_checkpoint(path, a.device)
        key = f"{info['architecture']}-seed{info['seed']}"
        if key in models:
            p.error(f'Duplicate model identifier: {key}')
        models[key] = model
        checkpoints[key] = dict(path=str(path), sha256=digest(path), epoch=info['epoch'],
                                validation_accuracy=info['validation_accuracy'])
    dataset = datasets.CIFAR10(a.data, train=False, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(a.seed))[:a.samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=a.batch_size, shuffle=False, num_workers=0)
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', config={k: [str(v) for v in value] if k == 'checkpoints' else str(value) if isinstance(value, Path) else value for k, value in vars(a).items()},
                    environment=environment(), checkpoints=checkpoints, test_indices=indices)
    write_json(a.output / 'manifest.json', manifest)
    groups, example_saved, started = defaultdict(list), False, time.time()
    fields = ['index', 'label', 'source', 'target', 'attack', 'source_clean', 'target_clean', 'source_adv', 'target_adv', 'linf', 'l2']
    with (a.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        offset = 0
        for x, y in loader:
            batch_indices = indices[offset:offset+len(y)]
            x, y = x.to(a.device), y.to(a.device)
            with torch.no_grad():
                clean = {}
                for name, model in models.items():
                    logits = model(x)
                    if logits.shape != (len(y), 10) or not torch.isfinite(logits).all():
                        raise ValueError('Models must return finite CIFAR-10 logits')
                    clean[name] = logits.argmax(1)
            for source, model in models.items():
                for attack in a.attacks:
                    adv = generate(model, x, y, attack, a)
                    linf, l2 = validate_perturbations(x, adv, attack, a.epsilon, a.l2_budget)
                    with torch.no_grad():
                        attacked = {name: net(adv).argmax(1) for name, net in models.items()}
                    if not example_saved and attack not in ('clean', 'noise'):
                        torch.save(dict(clean=x[:1].cpu(), adversarial=adv[:1].cpu(), label=y[:1].cpu(),
                                        index=batch_indices[0], source=source, attack=attack,
                                        clean_predictions={k: int(v[0]) for k, v in clean.items()},
                                        adversarial_predictions={k: int(v[0]) for k, v in attacked.items()}), a.output / 'example.pt')
                        example_saved = True
                    for target in models:
                        for i, index in enumerate(batch_indices):
                            row = dict(index=index, label=int(y[i]), source=source, target=target, attack=attack,
                                       source_clean=int(clean[source][i]), target_clean=int(clean[target][i]),
                                       source_adv=int(attacked[source][i]), target_adv=int(attacked[target][i]),
                                       linf=float(linf[i]), l2=float(l2[i]))
                            writer.writerow(row)
                            groups[source, target, attack].append(row)
            offset += len(y)
            print(f'Evaluated {offset}/{a.samples}', flush=True)
    results = [dict(source=s, target=t, attack=k, **summarize(rows)) for (s, t, k), rows in groups.items()]
    write_json(a.output / 'summary.json', results)
    manifest.update(status='complete', elapsed_seconds=time.time()-started,
                    predictions_sha256=digest(a.output / 'predictions.csv'))
    write_json(a.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
