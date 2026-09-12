# Submission readiness review

The nine checkpoints, three full-test $L_\infty$ budget sweeps, and planned PGD convergence checks are complete. The empirical record supports a focused workshop paper about directionality, budget dependence, and source-success conditioning. Pilot measurements remain excluded from the manuscript.

## Defensible findings

- All models use CIFAR-10 labels, ten-class heads, differentiable in-model normalization, and raw-pixel perturbation budgets.
- Existing clean errors are excluded from PTR and CTR. Every per-run table prints exact numerators and denominators.
- Transfer is strongly directional and stable in ordering across three training seeds.
- Transfer changes substantially from $2/255$ to $8/255$.
- PTR and CTR separate most clearly when VGG16 source ASR is imperfect at $2/255$.
- Stronger PGD source optimization does not monotonically increase transfer for every direction.
- The primary 40-step/five-restart PGD result is close to the 100-step/five-restart endpoint, including the seed-0 $2/255$ check.

These are descriptive results for the stated architectures, training procedures, and CIFAR-10 threat model. They do not establish architecture-only causality or general behavior across datasets and model families.

## Work remaining before submission

1. Run `transferlab.study_report --verify-predictions` to regenerate all cross-seed tables from checksum-verified per-example files.
2. Run the test suite and compile the revised LaTeX. Inspect the PDF for table width, references, page count, and accidental overflow.
3. Select the workshop and replace the generic article layout with its official template. Check anonymity, page limit, deadline, artifact rules, and required assistance disclosure.
4. Freeze a clean release commit or tag. The completed manifests record `git_dirty: true`; retain their source-file hashes and archive the exact executed source snapshot.
5. Archive checkpoints, histories, manifests, per-example CSVs, generated tables, and commands outside Git if size requires it. Publish immutable checksums and a stable artifact link.
6. Create one primary figure showing the six directional PGD curves over $2/255$, $4/255$, and $8/255$, with seed points or ranges visible. A second compact figure may show PTR versus CTR at $2/255$.
7. Have the author verify every manuscript number against the generated summary and confirm affiliation, contact address, coauthorship, acknowledgments, disclosure text, and repository license.

## Scientific limits reviewers may raise

- Three training seeds permit descriptive variability estimates but weak population-level inference.
- Only three conventional CNN families and one dataset are studied.
- VGG16 uses a validation-selected learning rate after the common recipe failed.
- Random-noise controls use separate deterministic draws for nominal source rows.
- The study does not include transformers, robust models, targeted attacks, AutoAttack, or an $L_2$ threat model.
- The contribution is measurement and reporting practice supported by a compact empirical case study, rather than a new attack or defense.

The strongest submission presents these limits directly and keeps the claim narrow: transfer rankings are conditional on direction, budget, source success, and the evaluated population.
