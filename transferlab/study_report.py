"""Aggregate the completed three-seed CIFAR-10 study.

This module never trains a model or generates an attack. It reads completed run
directories, validates their protocol fields, and writes publication tables.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from .report import verified_results


ARCHITECTURES = ("resnet18", "vgg16", "mobilenet_v2")
ATTACKS = ("noise", "fgsm", "pgd")
BUDGET_RUNS = {
    2: "linf-2of255-seed{seed}",
    4: "linf-4of255-seed{seed}",
    8: "primary-linf-seed{seed}",
}


def percent(value):
    return 100.0 * value


def mean_range(values):
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def load_run(path, expected_seed, expected_budget, verify_predictions):
    manifest_path = path / "manifest.json"
    summary_path = path / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing manifest or summary in {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("pilot"):
        raise ValueError(f"{path} is not a completed non-pilot run")
    config = manifest["config"]
    if config.get("samples") != 10000 or config.get("seed") != 0:
        raise ValueError(f"{path} does not use the full test set with evaluation seed 0")
    if abs(config.get("epsilon", -1) - expected_budget / 255) > 1e-10:
        raise ValueError(f"Unexpected epsilon in {path}")
    if abs(config.get("step_size", -1) - expected_budget / (4 * 255)) > 1e-10:
        raise ValueError(f"Unexpected PGD step size in {path}")
    if config.get("steps") != 40 or config.get("restarts") != 5:
        raise ValueError(f"{path} is not a 40-step, five-restart run")
    required = set(ATTACKS)
    if expected_budget == 8:
        required.update({"clean", "noise"})
    if not required.issubset(config.get("attacks", [])):
        raise ValueError(f"Required attacks are absent from {path}")
    checkpoint_names = set(manifest["checkpoints"])
    expected_names = {f"{name}-seed{expected_seed}" for name in ARCHITECTURES}
    if checkpoint_names != expected_names:
        raise ValueError(f"Checkpoint trio does not match seed {expected_seed} in {path}")
    rows = verified_results(path, manifest) if verify_predictions else json.loads(summary_path.read_text())
    return manifest, rows


def lookup(rows, source, target, attack="pgd"):
    matches = [r for r in rows if r["source"] == source and r["target"] == target
               and r["attack"] == attack]
    if len(matches) != 1:
        raise ValueError(f"Expected one {source}->{target} {attack} row, found {len(matches)}")
    return matches[0]


def load_sensitivity(path, expected_seed, steps, restarts, epsilon, verify_predictions):
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["config"]
    if manifest.get("status") != "complete" or manifest.get("pilot"):
        raise ValueError(f"{path} is not complete")
    expected = {"samples": 10000, "seed": 0, "steps": steps, "restarts": restarts}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unexpected sensitivity configuration in {path}")
    if (abs(config.get("epsilon", -1) - epsilon) > 1e-10
            or abs(config.get("step_size", -1) - epsilon / 4) > 1e-10
            or config.get("attacks") != ["pgd"]):
        raise ValueError(f"Unexpected sensitivity attack in {path}")
    names = set(manifest["checkpoints"])
    if names != {f"{a}-seed{expected_seed}" for a in ARCHITECTURES}:
        raise ValueError(f"Sensitivity checkpoints do not match seed {expected_seed}")
    return verified_results(path, manifest) if verify_predictions else json.loads(
        (path / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser(description="Aggregate the completed three-seed study")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/study"))
    parser.add_argument(
        "--verify-predictions",
        action="store_true",
        help="recompute every summary from checksum-verified prediction CSVs",
    )
    args = parser.parse_args()

    loaded = {}
    provenance = []
    for budget, pattern in BUDGET_RUNS.items():
        for seed in range(3):
            path = args.runs / pattern.format(seed=seed)
            manifest, rows = load_run(path, seed, budget, args.verify_predictions)
            loaded[budget, seed] = rows
            provenance.append({
                "run": str(path),
                "commit": manifest["environment"].get("git_commit"),
                "dirty": manifest["environment"].get("git_dirty"),
                "predictions_sha256": manifest.get("predictions_sha256"),
            })

    seed_rows = []
    clean = {a: [] for a in ARCHITECTURES}
    for seed in range(3):
        rows = loaded[8, seed]
        for source_arch in ARCHITECTURES:
            source = f"{source_arch}-seed{seed}"
            diagonal = lookup(rows, source, source, "clean")
            clean[source_arch].append(percent(diagonal["source_clean_accuracy"]["rate"]))
        for budget in BUDGET_RUNS:
            rows = loaded[budget, seed]
            for attack in ATTACKS:
                for source_arch in ARCHITECTURES:
                    for target_arch in ARCHITECTURES:
                        source = f"{source_arch}-seed{seed}"
                        target = f"{target_arch}-seed{seed}"
                        row = lookup(rows, source, target, attack)
                        seed_rows.append({
                            "budget_over_255": budget,
                            "attack": attack,
                            "seed": seed,
                            "source": source_arch,
                            "target": target_arch,
                            "source_asr_percent": percent(row["source_asr"]["rate"]),
                            "source_asr_numerator": row["source_asr"]["successes"],
                            "source_asr_denominator": row["source_asr"]["total"],
                            "ptr_percent": percent(row["pair_transfer"]["rate"]),
                            "ptr_numerator": row["pair_transfer"]["successes"],
                            "ptr_denominator": row["pair_transfer"]["total"],
                            "ctr_percent": percent(row["conditional_transfer"]["rate"]),
                            "ctr_numerator": row["conditional_transfer"]["successes"],
                            "ctr_denominator": row["conditional_transfer"]["total"],
                        })

    aggregates = {}
    for budget in BUDGET_RUNS:
        for attack in ATTACKS:
            for source in ARCHITECTURES:
                for target in ARCHITECTURES:
                    selected = [r for r in seed_rows if r["budget_over_255"] == budget
                                and r["attack"] == attack and r["source"] == source
                                and r["target"] == target]
                    aggregates[f"{budget}:{attack}:{source}:{target}"] = {
                        "ptr_percent": mean_range([r["ptr_percent"] for r in selected]),
                        "ctr_percent": mean_range([r["ctr_percent"] for r in selected]),
                        "source_asr_percent": mean_range(
                            [r["source_asr_percent"] for r in selected]),
                    }

    sensitivity = []
    for seed in range(3):
        primary = loaded[8, seed]
        strong_path = args.runs / f"pgd-sensitivity-seed{seed}-steps100-restarts5"
        strong = load_sensitivity(strong_path, seed, 100, 5, 8 / 255,
                                  args.verify_predictions)
        for source_arch in ARCHITECTURES:
            for target_arch in ARCHITECTURES:
                if source_arch == target_arch:
                    continue
                source, target = f"{source_arch}-seed{seed}", f"{target_arch}-seed{seed}"
                baseline_row, strong_row = lookup(primary, source, target), lookup(strong, source, target)
                sensitivity.append({
                    "budget_over_255": 8,
                    "seed": seed,
                    "source": source_arch,
                    "target": target_arch,
                    "baseline_ptr_percent": percent(baseline_row["pair_transfer"]["rate"]),
                    "strong_ptr_percent": percent(strong_row["pair_transfer"]["rate"]),
                    "difference_points": percent(strong_row["pair_transfer"]["rate"]
                                                 - baseline_row["pair_transfer"]["rate"]),
                })
    low_strong = load_sensitivity(
        args.runs / "pgd-2of255-sensitivity-seed0-steps100-restarts5",
        0, 100, 5, 2 / 255, args.verify_predictions,
    )
    for source_arch in ARCHITECTURES:
        for target_arch in ARCHITECTURES:
            if source_arch == target_arch:
                continue
            source, target = f"{source_arch}-seed0", f"{target_arch}-seed0"
            baseline_row = lookup(loaded[2, 0], source, target)
            strong_row = lookup(low_strong, source, target)
            sensitivity.append({
                "budget_over_255": 2,
                "seed": 0,
                "source": source_arch,
                "target": target_arch,
                "baseline_ptr_percent": percent(baseline_row["pair_transfer"]["rate"]),
                "strong_ptr_percent": percent(strong_row["pair_transfer"]["rate"]),
                "difference_points": percent(strong_row["pair_transfer"]["rate"]
                                             - baseline_row["pair_transfer"]["rate"]),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "seed_level.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_rows)
    with (args.output / "sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sensitivity[0].keys())
        writer.writeheader()
        writer.writerows(sensitivity)

    payload = {
        "clean_accuracy_percent": {a: mean_range(v) for a, v in clean.items()},
        "attack_aggregates": aggregates,
        "max_abs_8of255_strong_minus_primary_ptr_points": max(
            abs(r["difference_points"]) for r in sensitivity if r["budget_over_255"] == 8),
        "max_abs_2of255_strong_minus_primary_ptr_points": max(
            abs(r["difference_points"]) for r in sensitivity if r["budget_over_255"] == 2),
        "provenance": provenance,
    }
    (args.output / "study_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    labels = {"resnet18": "ResNet-18", "vgg16": "VGG16", "mobilenet_v2": "MobileNetV2"}
    lines = [
        "# Three-seed CIFAR-10 study", "",
        "Rates are percentages. Across-seed summaries treat the checkpoint seed as the unit; "
        "they do not pool repeated test images.", "", "## Clean accuracy", "",
        "| Model | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |", "|---|---:|---:|---:|---:|---:|",
    ]
    for arch in ARCHITECTURES:
        stats = mean_range(clean[arch])
        lines.append(f"| {labels[arch]} | {stats['values'][0]:.2f} | {stats['values'][1]:.2f} | "
                     f"{stats['values'][2]:.2f} | {stats['mean']:.2f} | {stats['sd']:.2f} |")
    lines.extend(["", "## Pair transfer by budget", ""])
    for budget in (2, 4, 8):
        lines.extend([
            f"### $L_\\infty={budget}/255$", "",
            "| Attack | Source | Target | Mean PTR | Seed range | Mean CTR | Mean source ASR |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for attack in ATTACKS:
            for source in ARCHITECTURES:
                for target in ARCHITECTURES:
                    if source == target:
                        continue
                    item = aggregates[f"{budget}:{attack}:{source}:{target}"]
                    ptr = item["ptr_percent"]
                    ctr = item["ctr_percent"]
                    asr = item["source_asr_percent"]
                    lines.append(f"| {attack.upper()} | {labels[source]} | {labels[target]} | "
                                 f"{ptr['mean']:.2f} | {ptr['min']:.2f}--{ptr['max']:.2f} | "
                                 f"{ctr['mean']:.2f} | {asr['mean']:.2f} |")
    lines.extend([
        "", "## Convergence checks", "",
        f"The largest absolute off-diagonal PTR change from 40 steps/five restarts to "
        f"100 steps/five restarts at 8/255 is "
        f"{payload['max_abs_8of255_strong_minus_primary_ptr_points']:.2f} percentage points.", "",
        f"For the seed-0 check at 2/255, the corresponding maximum is "
        f"{payload['max_abs_2of255_strong_minus_primary_ptr_points']:.2f} percentage points.", "",
        "Exact seed-level numerators and denominators are in `seed_level.csv`.", "",
        "## Provenance note", "",
        "The manifests preserve checkpoint hashes, prediction checksums, source-file hashes, and "
        "environment details. They also record a dirty working tree; archive the manifests and "
        "source snapshot with the submission.",
    ])
    (args.output / "study_summary.md").write_text("\n".join(lines) + "\n")
    print(args.output / "study_summary.md")


if __name__ == "__main__":
    main()
