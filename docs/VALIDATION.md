# Validation record

## Follow-up review and real-data pilot (2026-09-09)

Starting commit: `1ab8e349876a8f8630a9665bf73c03554421d3db`.

- Python 3.12.5, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 1.26.4, Foolbox 3.3.4, SciPy 1.17.1, Windows 11, NVIDIA RTX 4070.
- Final local `python -m pytest -q`: **36 passed in 9.54 seconds**, including the three CUDA model/input-gradient cases. One upstream Foolbox/SciPy deprecation warning remains. CPU CI skips CUDA cases when no GPU is available.
- `python -m pip check`: no broken requirements. Training, evaluation, and pilot CLI help execute successfully.
- The real-data three-architecture pilot completed in **82.3 seconds**, including training, checkpoint reload, all source-target combinations, clean/noise/FGSM/short PGD, zero-budget controls, short CW, report reconstruction, and figure generation. See [PILOT.md](PILOT.md).
- An independent check of 10,440 prediction rows confirmed pair/conditional numerators and denominators, exact identity predictions for clean/zero-budget controls, norm bounds, manifest indices, checkpoint hashes, train/validation disjointness, and selection by maximum validation accuracy.
- The saved figure was visually inspected. It correctly labels an already-misclassified clean example as a dog when its ground truth is horse, and does not claim that this is a successful attack.
- Fixes include nonfinite attack parameters/gradients/logits and training/validation failures, normalization metadata enforcement, singleton MobileNetV2 batches, complete report counts/norms, CSV-based report verification, explicit pilot labeling, hardware/source provenance, and compatible Python version constraints. GPU predictions are copied to CPU once per batch for CSV logging.
- `git diff --check` passes. The LaTeX source and existing PDF were unchanged; this follow-up did not rebuild the paper. No pilot results were inserted into it.

The pilot ran against the working-tree fixes on the starting commit. Its recorded source-file SHA-256 values identify those exact bytes, rather than falsely attributing execution to an already-clean commit. Full training, final test matrices, statistical analyses, and convergence studies remain unexecuted. The retained record below documents the earlier pass separately.

## Earlier methodological-correction validation

Validated locally on 2026-09-09 with Python 3.12, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, Foolbox 3.3.4, and pytest 8.3.5.

- `python -m pytest -q`: **19 passed**. One upstream Foolbox warning concerns SciPy's deprecated `ndimage.filters` import.
- Tests cover clean-error exclusion, distinct conditional denominators, undefined rates, Wilson interval boundaries, attack pixel/norm bounds including CW, zero-budget identity, FGSM direction, PGD source success on a toy classifier, ten-class logits and input gradients for all three architectures, rejection of incompatible checkpoints, safe checkpoint serialization round-trip, and a synthetic end-to-end evaluation/report run.
- Training and evaluation command-line help execute successfully.
- `make -C paper` compiles the revised PDF. The four rendered pages were inspected for layout and legibility. No overfull boxes or undefined references were reported.
- `git diff --check` passes.

These checks do not establish CIFAR-10 clean accuracy or empirical adversarial transfer. Full model training, real-data attack evaluation, CUDA execution, and attack convergence/sensitivity studies have not been run. The synthetic end-to-end fixture is explicitly not a benchmark result. CI configuration is included; local results must not be represented as a completed remote CI run.
