import csv

from transferlab.common import digest
from transferlab.study_report import conditioning_rows


def test_conditioning_decomposition_uses_pairwise_population(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    predictions = run / "predictions.csv"
    fieldnames = [
        "index", "label", "source", "target", "attack", "source_clean",
        "target_clean", "source_adv", "target_adv", "linf", "l2",
    ]
    rows = [
        # Two source successes, one of which transfers.
        [0, 0, "resnet18-seed0", "vgg16-seed0", "pgd", 0, 0, 1, 1, 0.1, 0.2],
        [1, 0, "resnet18-seed0", "vgg16-seed0", "pgd", 0, 0, 1, 0, 0.1, 0.2],
        # Two source failures, one of which fools the target.
        [2, 0, "resnet18-seed0", "vgg16-seed0", "pgd", 0, 0, 0, 1, 0.1, 0.2],
        [3, 0, "resnet18-seed0", "vgg16-seed0", "pgd", 0, 0, 0, 0, 0.1, 0.2],
        # Target-clean error: excluded from the pairwise population.
        [4, 0, "resnet18-seed0", "vgg16-seed0", "pgd", 0, 1, 1, 1, 0.1, 0.2],
    ]
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    result = conditioning_rows(
        run, {"predictions_sha256": digest(predictions)}, seed=0)[0]
    assert result["eligible"] == 4
    assert result["source_success"] == 2
    assert result["failed_source_target_failure"] == 1
    assert result["pair_source_asr_percent"] == 50
    assert result["ptr_percent"] == 50
    assert result["ctr_percent"] == 50
    assert result["failed_source_target_error_percent"] == 50
    reconstructed = (
        result["pair_source_asr_percent"] / 100 * result["ctr_percent"]
        + (1 - result["pair_source_asr_percent"] / 100)
        * result["failed_source_target_error_percent"]
    )
    assert reconstructed == result["ptr_percent"]
