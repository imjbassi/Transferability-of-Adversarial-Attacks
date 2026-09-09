# Running the corrected project yourself

## Windows PowerShell setup

From your checkout of this repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m transferlab.pilot --output runs/pilot-01
```

The CUDA command targets the local NVIDIA machine. For CPU-only machines, replace `cu124` with `cpu`. These commands do not require PowerShell activation or changing execution policy. Do not use the system Python 3.14 with these pinned dependencies.

The pilot downloads CIFAR-10 on first use. Output directories must be new. If a run fails, inspect its stage log and use another directory when restarting; training resume is not implemented. `pilot.json` records each subprocess command, return code, and wall time. Per-model `history.json` records epoch times and validation accuracy. Evaluation directories contain manifests, predictions, checked reports, and an example figure.

## What the pilot establishes

All three model training paths, checkpoint serialization/loading, cross-model evaluation, clean/random controls, FGSM, short PGD, short CW, exact zero-budget checks, report reconstruction, and figure rendering execute on real CIFAR-10 data. It does not establish useful model accuracy, attack convergence, transfer rankings, or submission readiness.

The pilot uses two epochs on 2,048 training examples and 512 validation examples. These are prefixes of the original fixed split, with no overlap and no test-based checkpoint selection. It uses 256 test examples for Linf, 32 for zero-budget checks, and eight for CW. Undertraining can leave very few jointly correct examples; `undefined [0/0]` is then the correct output.

## Measured local pilot timing

On 2026-09-09, the complete CUDA pilot took **82.3 seconds** on the local RTX 4070, excluding environment installation and archive download (including extraction and subprocess startup). Allow approximately **2–5 minutes** for a repeat with the environment and data already present; competing GPU workloads can change this. Fresh setup required a 2.5 GB PyTorch wheel plus dependencies, and the official CIFAR-10 archive is approximately 163 MiB. Allow additional download/install time; the initial sequential dataset download was slow on this connection. See [the pilot validation record](PILOT.md).

## Full-study scheduling

Use the full training/evaluation commands in README.md and the complete matrix in EXPERIMENTS.md. Do not reuse pilot checkpoints for final tables. There are nine primary training runs (three architectures times three seeds), each with 200 epochs on 45,000 training examples. Pilot training performs only two epochs on 2,048 examples: the primary training exposure is about 2,197 times larger per model, with additional validation and different batch-size costs. This ratio is not a reliable wall-time prediction.

Before committing to a long run, time several full-data training epochs per architecture with the intended settings, using validation only. Estimate training time from seconds/epoch times 200 times three seeds, summed over architectures; allow for initialization and checkpoint I/O. Freeze the eventual recipe before evaluating test attacks. Primary PGD alone uses 40 steps and five restarts, versus five steps and one restart in the pilot. Full CW with 1,000 steps and nine searches is much more expensive than the eight-image wiring check. Budget these stages independently and expect hours to days rather than pilot-scale minutes; actual timing depends on hardware and competing workloads.

No automatic full-study launch is scheduled. The pilot command completes and returns control to you.
