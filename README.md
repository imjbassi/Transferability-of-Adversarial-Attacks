# Adversarial transfer on CIFAR-10

**Status: corrected research implementation and revised methodological draft. New CIFAR-10 experiments are required before an empirical submission.**

The earlier code evaluated ImageNet classifiers against CIFAR-10 labels and called unconditional target error a transfer success rate. Those results do not establish adversarial transfer. The previous claims of approximately 90% clean accuracy and near-100% transfer are not supported by the released artifacts. See [the audit](docs/AUDIT.md).

- [Revised paper (PDF)](paper/main.pdf)
- [Complete editable LaTeX manuscript](paper/main.tex)
- [Experiment and submission requirements](docs/EXPERIMENTS.md)
- [Validation record](docs/VALIDATION.md)

## What changed

The code trains ten-class CIFAR-10 models, selects checkpoints on a held-out validation split, and places normalization inside each differentiable model. Attacks operate on raw pixels in `[0,1]`. FGSM, PGD, identity, and random-noise controls share the same evaluation pipeline. Each source attack is reused across every target. Outputs include checkpoint hashes, test indices, actual predictions, norm checks, denominators, and Wilson intervals. The revised manuscript reports the completed three-seed, full-test $L_\infty$ study; the [small real-data pilot](docs/PILOT.md) remains execution validation only.

## Install

Python 3.10–3.12 is required by this pinned environment. Python 3.11 is used in CI. On Windows, use `py -3.12 -m venv .venv` if the default Python is newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. For CPU-only use, install the pinned CPU wheels before installing the project:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

For a CUDA machine, install a compatible build of the same PyTorch/torchvision versions. Full multi-seed training and iterative attacks are substantial compute workloads. The completed study used PyTorch 2.6.0+cu124 on an NVIDIA RTX 4070.

## Train

### Run a small execution pilot first

After installation, run:

```bash
python -m transferlab.pilot --output runs/pilot-01
```

This trains all three architectures for two epochs on 2,048 training images, selects checkpoints on 512 validation images, and evaluates 256 test images with clean/noise/FGSM/short PGD controls. It also runs a 32-image zero-budget check and an eight-image, five-step CW wiring check. Reduced subsets preserve the fixed split boundary. These deliberately undertrained checkpoints and weak attacks are **not final empirical evidence**. Each stage writes a log and timing to the output directory; choose a new directory for another run. Add `--device cpu` when CUDA is unavailable.

See [the run guide](docs/RUNNING.md) for Windows setup, measured timings, and the distinction between the pilot and full study, and [the submission critique](docs/SUBMISSION_REVIEW.md) for remaining work.

### Full training

This complete command sequence trains one checkpoint per architecture. It downloads CIFAR-10, uses 45,000 training examples and 5,000 validation examples, and never selects a checkpoint on test performance.

```bash
python -m transferlab.train --architecture resnet18 --seed 0 --output checkpoints/resnet18-0
python -m transferlab.train --architecture vgg16 --seed 0 --output checkpoints/vgg16-0
python -m transferlab.train --architecture mobilenet_v2 --seed 0 --output checkpoints/mobilenet_v2-0
```

Repeat with seeds 1 and 2 for the planned multi-seed study. Output directories must be new, preventing accidental overwrite of existing runs. Training defaults to 200 epochs. A shorter run can test execution but must not be presented as a final benchmark.

## Evaluate and make a table

```bash
python -m transferlab.evaluate --checkpoints checkpoints/resnet18-0/best.pt checkpoints/vgg16-0/best.pt checkpoints/mobilenet_v2-0/best.pt --attacks clean noise fgsm pgd --samples 10000 --seed 0 --epsilon 0.03137254901960784 --steps 40 --step-size 0.00784313725490196 --restarts 5 --output runs/linf-seed0
python -m transferlab.report runs/linf-seed0
python -m transferlab.figure runs/linf-seed0/example.pt --output runs/linf-seed0/example.png
```

For a pilot, use `--samples 1000` and a different output directory. Pilot results are not full-test results. For a zero-budget control, use `--epsilon 0 --attacks clean fgsm pgd`. The paper's completed empirical scope is $L_\infty$; an $L_2$ attack requires a separate protocol and must not be mixed into these tables.

## Aggregate the completed study

After placing the completed run directories under `runs`, verify every prediction checksum and generate the cross-seed publication tables:

```powershell
.\.venv\Scripts\python.exe -m transferlab.study_report --runs runs --output artifacts\study --verify-predictions
```

This reads existing results only. It writes a Markdown summary, a JSON summary, exact seed-level counts, and convergence comparisons under `artifacts\study`.

Create the two-panel publication figure:

```powershell
.\.venv\Scripts\python.exe -m transferlab.study_figure artifacts\study --output artifacts\study\transfer_summary.pdf
```

## Metrics

| Output | Numerator | Denominator |
|---|---|---|
| Source ASR | Source attacked errors among source-clean-correct inputs | Source-clean-correct inputs |
| Pair transfer | Target attacked errors among jointly clean-correct inputs | Source and target jointly clean-correct inputs |
| Conditional transfer | Target attacked errors among jointly clean-correct inputs where the source attack succeeded | Jointly clean-correct inputs where the source attack succeeded |
| Adversarial accuracy | Correct target predictions on attacked inputs | All evaluated inputs |

Rates in JSON are fractions, while generated Markdown tables use percentages. Empty denominators produce JSON `null`, not zero. Diagonal rows are white-box controls. A diagonal conditional-transfer rate of one is a consequence of conditioning, not evidence of black-box transfer. Wilson intervals condition on fixed checkpoints and do not account for training-seed variability.

## Artifact outputs

Each evaluation produces `manifest.json`, `predictions.csv`, `summary.json`, and, when an actual attack is selected, `example.pt`. The report command verifies the prediction checksum, recomputes summary metrics from CSV, and writes `table.md` with counts and norm diagnostics. The manifest is marked complete only after evaluation finishes. Shortened training and subset evaluations are labeled as pilots. The figure command labels the actual source predictions and scales the perturbation explicitly; it does not assume the selected example is successful.

The legacy filenames now delegate to the corrected CLI. For example, `python FGSM_transfer.py --help` describes required checkpoint arguments. Historical results and figures are preserved under `archive/original/` and are not validation evidence.

## Build the paper

```bash
make -C paper
```

A standard LaTeX installation with `pdflatex` is required. The manuscript uses a generic two-column format. Choose a workshop and adapt its official template, anonymization rules, and page limit before submission.

## Licensing

The previous README referred to a `LICENSE` file that was not present. No new license has been assigned in this revision; the repository owner must choose the intended terms before describing the project as open-source or granting reuse permissions.
