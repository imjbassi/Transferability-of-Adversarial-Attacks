import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .common import CLASSES


def main():
    p = argparse.ArgumentParser(description='Render the actual saved example, including prediction labels')
    p.add_argument('example', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = torch.load(a.example, map_location='cpu', weights_only=True)
    x, adv = data['clean'][0], data['adversarial'][0]
    delta = adv - x
    scale = float(delta.abs().max())
    images = [x, adv, delta / max(scale, 1e-12) / 2 + 0.5]
    source = data['source']
    titles = [f"Clean: {CLASSES[data['clean_predictions'][source]]}",
              f"Attacked: {CLASSES[data['adversarial_predictions'][source]]}",
              f'Perturbation (scaled)\nmax absolute = {scale:.5f}']
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.suptitle(f"Test index {data['index']} | true: {CLASSES[int(data['label'][0])]} | {source} | {data['attack']}", fontsize=10)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
