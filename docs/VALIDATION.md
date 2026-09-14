# Validation record

## Submission cleanup (2026-09-13)

The abstract now leads with the target-dependent denominator mismatch: conventional source ASR and PTR cannot reconstruct CTR. The conditioning result is stated narrowly—five directions are nearly mechanical, while VGG16-to-MobileNetV2 has the only material failed-source residual. The introduction lists three contributions, the redundant error-attribution discussion is reduced to two sentences, and the matched-learning-rate paragraph states exactly what the control licenses.

The primary $8/255$ PGD matrix reports the mean and minimum--maximum range over three training seeds; fixed-checkpoint Wilson intervals remain in the per-run artifacts. The conditioning table prints all pairwise-eligible and source-success counts for seeds 0/1/2. The paper restores the single-panel budget sweep with seed-range error bars and omits the duplicated conditioning panel. The conditioning decomposition was rechecked directly from the per-seed CSV: the five-direction mean-level maximum deviation from the zero-residual approximation is 0.27 points, and the individual-seed maximum is 0.40 points. The final PDF was compiled twice and all pages were visually inspected. There are no overfull boxes, undefined citations, or unresolved references. The test suite passes with **37 tests**; the sole warning remains the upstream Foolbox/SciPy deprecation.

## Conditioning and related-work revision (2026-09-12)

The checksum-verified per-example predictions were reanalyzed by splitting each pairwise-eligible set into source-success and source-failure strata. The generated `artifacts/study/conditioning_decomposition.csv` records exact counts for both strata and verifies

\[
\mathrm{PTR}_{st}=a_{st}\mathrm{CTR}_{st}+(1-a_{st})b_{st}.
\]

Here, \(a_{st}\) is source success on the pairwise-eligible set and \(b_{st}\) is target error among eligible examples on which the source attack failed. At $2/255$, the mean failed-source target-error rate is 0% in three directions, 0.25% for ResNet-18 to MobileNetV2, 0.68% for VGG16 to ResNet-18, and 2.83% for VGG16 to MobileNetV2. The last two values show that the zero-residual approximation is close but not exact; conventional source ASR also has a broader denominator than pairwise source success.

The manuscript now reports source ASR and the complete FGSM transfer matrix, positions directionality and source-optimization effects against prior work, and reduces the self-audit to a motivating paragraph. The aggregation figure now compares observed CTR with the zero-residual approximation and labels the measured failed-source residual. The full test suite passes with **37 tests**; the only warning is the documented upstream Foolbox import of SciPy's deprecated `ndimage.filters` namespace. MiKTeX 24.1 compiles the six-page manuscript twice with no overfull boxes, undefined citations, or unresolved references.

The learning-rate sensitivity check trained seed-0 ResNet-18 and MobileNetV2 checkpoints for 200 epochs at initial learning rate 0.01, matching VGG16. Their best validation accuracies were 94.78% and 91.36%; clean test accuracies were 93.82% and 90.90%. The full-test $8/255$ evaluation completed in 1,864.8 seconds. VGG16-to-ResNet-18 PTR was 95.98% versus 32.71% in reverse, and VGG16-to-MobileNetV2 was 93.12% versus 18.55% in reverse. The VGG asymmetry therefore survives matched initial learning rates, although individual matrix entries changed by as much as 25.93 points and the control has only one seed. `artifacts/study/lr_matched_sensitivity.csv` is generated from checksum-verified predictions.

## Post-experiment manuscript and aggregation update (2026-09-11)

The full three-seed $L_\infty$ experiment record was used to revise `paper/main.tex` and add `transferlab.study_report` and `transferlab.study_figure`. The final seed-0 $2/255$ convergence run completed in 5,894.1 seconds; relative to 40-step/five-restart PGD, 100 steps/five restarts changed every off-diagonal PTR by at most 0.27 percentage points and every CTR by at most 0.40 points.

The checksum-verifying aggregation completed successfully and recomputed all included summaries from the saved prediction CSVs. The publication figure was generated as PDF and 300-dpi PNG and visually inspected. The test suite passed with **36 tests** in 13.33 seconds; the one warning is the documented upstream Foolbox import of SciPy's deprecated `ndimage.filters` namespace. MiKTeX 24.1 compiled the revised six-page manuscript twice. The final log has no overfull boxes, undefined citations, or unresolved references. All six rendered pages were visually inspected for clipping, overlap, table readability, and figure legibility.

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
