"""Build the anonymous TMLR supplement and the public experiment record.

The generated ZIP files are placed in ``release-assets/`` and intentionally
ignored by Git. Run this script from the repository root after the paper and
aggregate study artifacts have been regenerated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release-assets"

EVALUATION_RUNS = [
    *(f"linf-2of255-seed{seed}" for seed in range(3)),
    *(f"linf-4of255-seed{seed}" for seed in range(3)),
    *(f"primary-linf-seed{seed}" for seed in range(3)),
    "pgd-2of255-sensitivity-seed0-steps100-restarts5",
    "pgd-sensitivity-seed0-steps10-restarts1",
    "pgd-sensitivity-seed0-steps40-restarts1",
    "pgd-sensitivity-seed0-steps100-restarts1",
    "pgd-sensitivity-seed0-steps100-restarts5",
    "pgd-sensitivity-seed1-steps10-restarts1",
    "pgd-sensitivity-seed1-steps100-restarts5",
    "pgd-sensitivity-seed2-steps10-restarts1",
    "pgd-sensitivity-seed2-steps100-restarts5",
    "lr-matched-sensitivity/linf-8of255-seed0-lr001",
]

TRAINING_RUNS = [
    *(f"full-study/checkpoints/resnet18-seed{seed}" for seed in range(3)),
    *(f"vgg-tuning/vgg16-seed{seed}-lr001" for seed in range(3)),
    *(f"mobilenet-training/mobilenet_v2-seed{seed}" for seed in range(3)),
    "lr-matched-sensitivity/resnet18-seed0-lr001",
    "lr-matched-sensitivity/mobilenet_v2-seed0-lr001",
]

SOURCE_FILES = [
    "pyproject.toml",
    *(str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / "transferlab").glob("*.py"))),
    *(str(path.relative_to(ROOT)).replace("\\", "/") for path in sorted((ROOT / "tests").glob("test_*.py"))),
]

TMLR_SOURCE_FILES = [
    "main.tex",
    "main.bib",
    "tmlr.sty",
    "tmlr.bst",
    "fancyhdr.sty",
    "transfer_summary.pdf",
]

ANONYMOUS_README = """# Anonymous reproducibility supplement

This package accompanies the double-blind TMLR submission
"Conditioning and Directionality in Adversarial Transfer on CIFAR-10."

Contents:
- `source/`: corrected implementation and tests;
- `artifacts/study/`: aggregate tables and figures used by the paper;
- `runs/`: per-example predictions, run summaries, and anonymized manifests;
- `training_records/`: seed-level training histories and anonymized metadata.

The per-example records cover every final evaluation reported in the paper.
Model checkpoint binaries are omitted to remain below the venue's 100 MB
supplement limit. Their SHA-256 hashes remain in the manifests, and the public
experiment-record archive will contain the binaries after double-blind review.

The manifests truthfully retain `git_dirty: true` from execution. Commit values
are redacted only for double-blind review; source-file and checkpoint hashes are
unchanged. See `source/pyproject.toml` for dependencies and run `pytest` from
`source/` to execute the test suite.
"""

PUBLIC_README = """# Experiment records

This archive contains the final experiment records for "Conditioning and
Directionality in Adversarial Transfer on CIFAR-10": aggregate study artifacts,
per-example predictions, manifests, training histories, and the eleven model
checkpoints used for the primary study and matched-learning-rate control.

The nine primary checkpoints cover ResNet-18, VGG16, and MobileNetV2 at seeds
0, 1, and 2. Two additional seed-0 checkpoints support the matched-learning-rate
control. The recorded manifests truthfully report that execution occurred on a
dirty working tree and retain the executed-source hashes, checkpoint hashes,
prediction hashes, environment, and exact sample indices needed for audit.

The accompanying source release is archived separately under the project DOI.
"""


def _arc(path: Path, prefix: str = "") -> str:
    relative = str(path.relative_to(ROOT)).replace("\\", "/")
    return f"{prefix}{relative}"


def _sanitized_json(path: Path) -> bytes:
    data = json.loads(path.read_text(encoding="utf-8"))

    def redact(value):
        if isinstance(value, dict):
            return {
                key: (
                    "withheld_for_double_blind_review"
                    if key in {"git_commit", "commit"}
                    else redact(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return (json.dumps(redact(data), indent=2) + "\n").encode("utf-8")


def _write_tree(
    archive: zipfile.ZipFile,
    source: Path,
    destination: str,
    *,
    anonymize_json: bool,
    exclude_suffixes: set[str] | None = None,
) -> None:
    excluded = exclude_suffixes or set()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix in excluded:
            continue
        target = f"{destination}/{path.relative_to(source).as_posix()}"
        if anonymize_json and path.suffix == ".json":
            archive.writestr(target, _sanitized_json(path))
        else:
            archive.write(path, target)


def build_anonymous_supplement(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("README.md", ANONYMOUS_README)
        for relative in SOURCE_FILES:
            archive.write(ROOT / relative, f"source/{relative}")
        _write_tree(
            archive,
            ROOT / "artifacts" / "study",
            "artifacts/study",
            anonymize_json=True,
        )
        for relative in EVALUATION_RUNS:
            source = ROOT / "runs" / relative
            _write_tree(
                archive,
                source,
                f"runs/{relative}",
                anonymize_json=True,
                exclude_suffixes={".pt"},
            )
        for relative in TRAINING_RUNS:
            source = ROOT / "runs" / relative
            _write_tree(
                archive,
                source,
                f"training_records/{relative}",
                anonymize_json=True,
                exclude_suffixes={".pt"},
            )


def build_public_record(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.md", PUBLIC_README)
        archive.write(ROOT / "LICENSE", "LICENSE")
        _write_tree(
            archive,
            ROOT / "artifacts" / "study",
            "artifacts/study",
            anonymize_json=False,
        )
        for relative in EVALUATION_RUNS:
            _write_tree(
                archive,
                ROOT / "runs" / relative,
                f"runs/{relative}",
                anonymize_json=False,
            )
        for relative in TRAINING_RUNS:
            _write_tree(
                archive,
                ROOT / "runs" / relative,
                f"runs/{relative}",
                anonymize_json=False,
            )


def build_tmlr_source(path: Path) -> None:
    source = ROOT / "paper" / "tmlr"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in TMLR_SOURCE_FILES:
            archive.write(source / name, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    anonymous = OUTPUT / "tmlr-anonymous-supplement.zip"
    public = OUTPUT / "adversarial-transfer-experiment-records-v1.0.1.zip"
    tmlr_source = OUTPUT / "tmlr-submission-source.zip"
    build_anonymous_supplement(anonymous)
    build_public_record(public)
    build_tmlr_source(tmlr_source)
    checksums = OUTPUT / "SHA256SUMS.txt"
    checksums.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in (anonymous, public, tmlr_source)
        ),
        encoding="utf-8",
    )
    zenodo_checksum = OUTPUT / "ZENODO_SHA256.txt"
    zenodo_checksum.write_text(
        f"{_sha256(public)}  {public.name}\n",
        encoding="utf-8",
    )
    for path in (anonymous, public, tmlr_source, checksums, zenodo_checksum):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
