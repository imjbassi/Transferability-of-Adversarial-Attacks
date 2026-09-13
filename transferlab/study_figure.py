"""Create the publication figure from an aggregated study report."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "resnet18": "ResNet-18",
    "vgg16": "VGG16",
    "mobilenet_v2": "MobileNetV2",
}
COLORS = {
    "resnet18": "#0072B2",
    "vgg16": "#D55E00",
    "mobilenet_v2": "#009E73",
}
MARKERS = {"resnet18": "o", "vgg16": "s", "mobilenet_v2": "^"}


def main():
    parser = argparse.ArgumentParser(description="Plot the completed transfer study")
    parser.add_argument("study", type=Path, help="directory produced by transferlab.study_report")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.study / "transfer_summary.pdf"

    with (args.study / "seed_level.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "budget_over_255", "attack", "seed", "source", "target", "ptr_percent",
        "ctr_percent",
    }
    if not rows or not required.issubset(rows[0]):
        parser.error("seed_level.csv is empty or has an incompatible schema")
    pgd = [r for r in rows if r["attack"] == "pgd" and r["source"] != r["target"]]

    grouped = defaultdict(list)
    for row in pgd:
        key = int(row["budget_over_255"]), row["source"], row["target"]
        grouped[key].append(float(row["ptr_percent"]))
    expected = {(budget, source, target) for budget in (2, 4, 8) for source in LABELS
                for target in LABELS if source != target}
    if set(grouped) != expected or any(len(values) != 3 for values in grouped.values()):
        parser.error("expected three seeds for every off-diagonal PGD budget/pair")

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10})
    fig, axis = plt.subplots(figsize=(4.7, 3.05), constrained_layout=True)
    line_styles = {"resnet18": "-", "vgg16": "--", "mobilenet_v2": ":"}
    for source in LABELS:
        for target in LABELS:
            if source == target:
                continue
            means, lows, highs = [], [], []
            for budget in (2, 4, 8):
                values = grouped[budget, source, target]
                means.append(statistics.mean(values))
                lows.append(means[-1] - min(values))
                highs.append(max(values) - means[-1])
            label = f"{LABELS[source]} $\\to$ {LABELS[target]}"
            axis.errorbar(
                (2, 4, 8), means, yerr=(lows, highs), label=label,
                color=COLORS[source], linestyle=line_styles[target], marker=MARKERS[target],
                linewidth=1.25, markersize=4, capsize=2,
            )
    axis.set_title("PGD transfer versus budget")
    axis.set_xlabel(r"$L_\infty$ budget ($1/255$)")
    axis.set_ylabel("Pair transfer rate (%)")
    axis.set_xticks((2, 4, 8))
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(fontsize=6.4, frameon=False, ncol=1, loc="upper left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
