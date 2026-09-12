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
SHORT_LABELS = {"resnet18": "R", "vgg16": "V", "mobilenet_v2": "M"}
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
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.05), constrained_layout=True)
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
            left.errorbar(
                (2, 4, 8), means, yerr=(lows, highs), label=label,
                color=COLORS[source], linestyle=line_styles[target], marker=MARKERS[target],
                linewidth=1.25, markersize=4, capsize=2,
            )
    left.set_title("PGD transfer versus budget")
    left.set_xlabel(r"$L_\infty$ budget ($1/255$)")
    left.set_ylabel("Pair transfer rate (%)")
    left.set_xticks((2, 4, 8))
    left.set_ylim(0, 100)
    left.grid(alpha=0.25, linewidth=0.6)
    left.legend(fontsize=6.4, frameon=False, ncol=1, loc="upper left")

    with (args.study / "conditioning_decomposition.csv").open(newline="") as handle:
        decomposition = list(csv.DictReader(handle))
    decomposition_required = {
        "seed", "source", "target", "pair_source_asr_percent", "ptr_percent",
        "ctr_percent", "failed_source_target_error_percent",
    }
    if not decomposition or not decomposition_required.issubset(decomposition[0]):
        parser.error("conditioning_decomposition.csv is empty or incompatible")
    pair_rows = defaultdict(list)
    for row in decomposition:
        pair_rows[row["source"], row["target"]].append(row)
    pair_order = (
        ("resnet18", "vgg16"), ("resnet18", "mobilenet_v2"),
        ("vgg16", "resnet18"), ("vgg16", "mobilenet_v2"),
        ("mobilenet_v2", "resnet18"), ("mobilenet_v2", "vgg16"),
    )
    positions = range(len(pair_order))
    names, derived, observed, failed_target = [], [], [], []
    for pair in pair_order:
        values = pair_rows[pair]
        names.append(f"{SHORT_LABELS[pair[0]]}$\\to${SHORT_LABELS[pair[1]]}")
        derived.append(statistics.mean(
            float(r["ptr_percent"]) / (float(r["pair_source_asr_percent"]) / 100)
            for r in values))
        observed.append(statistics.mean(float(r["ctr_percent"]) for r in values))
        failed_target.append(statistics.mean(
            float(r["failed_source_target_error_percent"]) for r in values))
    width = 0.38
    right.bar([p - width / 2 for p in positions], derived, width,
              label=r"PTR/$a_{st}$ ($b_{st}=0$)", color="#56B4E9")
    right.bar([p + width / 2 for p in positions], observed, width,
              label="Observed CTR", color="#E69F00")
    for position, height, residual in zip(positions, observed, failed_target):
        right.text(position, max(height, derived[position]) + 0.45, f"$b$={residual:.2f}",
                   ha="center", va="bottom", fontsize=6.5)
    right.set_title(r"Conditioning identity at $2/255$")
    right.set_ylabel("Conditional transfer rate (%)")
    right.set_xticks(list(positions), names, fontsize=8)
    right.set_ylim(0, max(derived) * 1.22)
    right.grid(axis="y", alpha=0.25, linewidth=0.6)
    right.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
