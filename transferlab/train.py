import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .common import CLASSES, environment, seed_all, write_json
from .models import CIFARModel, NAMES, SPEC


def main():
    p = argparse.ArgumentParser(description='Train CIFAR-10 models without test-set checkpoint selection')
    p.add_argument('--architecture', choices=NAMES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', default='data')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=1729)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0:
        p.error('epochs, batch size, and learning rate must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    seed_all(a.seed)
    augmentation = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_data = datasets.CIFAR10(a.data, train=True, download=True, transform=augmentation)
    validation_data = datasets.CIFAR10(a.data, train=True, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(a.split_seed)).tolist()
    train_indices, val_indices = indices[5000:], indices[:5000]
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(validation_data, val_indices), batch_size=a.batch_size, num_workers=0)
    model = CIFARModel(a.architecture).to(a.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)
    criterion = nn.CrossEntropyLoss()
    history, best, started = [], -1., time.time()
    metadata = dict(spec=SPEC, architecture=a.architecture, classes=CLASSES,
                    seed=a.seed, split_seed=a.split_seed, train_indices=train_indices,
                    validation_indices=val_indices, environment=environment(),
                    config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    for epoch in range(a.epochs):
        model.train()
        total_loss = 0.
        for x, y in train_loader:
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += model(x.to(a.device)).argmax(1).cpu().eq(y).sum().item()
        accuracy = correct / len(val_indices)
        row = dict(epoch=epoch+1, train_loss=total_loss/len(train_indices), validation_accuracy=accuracy,
                   learning_rate=optimizer.param_groups[0]['lr'], elapsed_seconds=time.time()-started)
        history.append(row)
        print(row, flush=True)
        if accuracy > best:
            best = accuracy
            torch.save(dict(metadata, state_dict=model.state_dict(), epoch=epoch+1,
                            validation_accuracy=accuracy), a.output / 'best.pt')
        scheduler.step()
        write_json(a.output / 'history.json', history)
    write_json(a.output / 'metadata.json', metadata)


if __name__ == '__main__':
    main()
