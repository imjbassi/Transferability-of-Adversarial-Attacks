# Validation record

Validated locally on 2026-09-09 with Python 3.12, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, Foolbox 3.3.4, and pytest 8.3.5.

- `python -m pytest -q`: **19 passed**. One upstream Foolbox warning concerns SciPy's deprecated `ndimage.filters` import.
- Tests cover clean-error exclusion, distinct conditional denominators, undefined rates, Wilson interval boundaries, attack pixel/norm bounds including CW, zero-budget identity, FGSM direction, PGD source success on a toy classifier, ten-class logits and input gradients for all three architectures, rejection of incompatible checkpoints, safe checkpoint serialization round-trip, and a synthetic end-to-end evaluation/report run.
- Training and evaluation command-line help execute successfully.
- `make -C paper` compiles the revised PDF. The four rendered pages were inspected for layout and legibility. No overfull boxes or undefined references were reported.
- `git diff --check` passes.

These checks do not establish CIFAR-10 clean accuracy or empirical adversarial transfer. Full model training, real-data attack evaluation, CUDA execution, and attack convergence/sensitivity studies have not been run. The synthetic end-to-end fixture is explicitly not a benchmark result. CI configuration is included; local results must not be represented as a completed remote CI run.
