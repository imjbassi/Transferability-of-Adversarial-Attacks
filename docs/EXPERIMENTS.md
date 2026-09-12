# Experiment specification and completed study

## Execution status (2026-09-11)

The nine planned checkpoints and the full 10,000-image $L_\infty$ evaluations are complete. The completed record includes:

- three ResNet-18 checkpoints trained at learning rate 0.1;
- three MobileNetV2 checkpoints trained at learning rate 0.1;
- three VGG16 checkpoints trained at learning rate 0.01 after the common 0.1 recipe failed and 0.01 was selected on validation data;
- clean, random-noise, FGSM, and 40-step/five-restart PGD evaluation at $8/255$;
- random-noise, FGSM, and 40-step/five-restart PGD evaluation at $2/255$ and $4/255$;
- replicated 10-step/one-restart and 100-step/five-restart PGD endpoints at $8/255$;
- a 100-step/five-restart seed-0 convergence check at $2/255$.

The empirical paper is scoped to $L_\infty$. CW-$L_2$ is not a required remaining run and must not be presented as completed evidence.

## Primary study

1. Train ResNet-18, adapted VGG16, and MobileNetV2 for each seed in `{0,1,2}`. Keep split seed 1729 constant. Record validation-selected checkpoints, training histories, hardware, elapsed time, and versions.
2. Freeze training choices before examining test attacks. If the recipe undertrains an architecture, tune on validation data, document the change, and rerun its planned seeds. Do not assert a target clean accuracy in advance.
3. Evaluate all 10,000 test examples using identical indices and evaluation seed 0 for each checkpoint trio. Include all nine source-target entries. The diagonal measures source attack strength; six off-diagonal entries measure transfer.
4. Run identity and random-noise controls plus FGSM and PGD at L-infinity budgets 0, 2/255, 4/255, and 8/255. Record explicit step sizes. Run PGD with 10, 40, and 100 steps and 1 and 5 restarts to examine sensitivity. Primary PGD uses 40 steps and 5 restarts at 8/255.
5. Generate tables using `transferlab.report`. Show source ASR, pair transfer, conditional transfer, both clean accuracies, raw counts, and norm diagnostics. Preserve prediction CSVs and manifests alongside final tables.

## Final aggregation

Generate the cross-seed publication tables from the repository root:

```powershell
.\.venv\Scripts\python.exe -m transferlab.study_report --runs runs --output artifacts\study --verify-predictions
```

The verification option recomputes every included run from its checksum-verified prediction CSV, so it will take longer than reading the saved summaries. It does not train models or generate attacks. Outputs are `study_summary.md`, `study_summary.json`, `seed_level.csv`, and `sensitivity.csv`.

Create the publication figure from those verified tables:

```powershell
.\.venv\Scripts\python.exe -m transferlab.study_figure artifacts\study --output artifacts\study\transfer_summary.pdf
```

This writes a vector PDF and a 300-dpi PNG. The left panel shows seed ranges for all six directional PGD budget curves; the right panel compares PTR and CTR at $2/255$.

## Statistical interpretation

Show each training seed separately and summarize across seed-level estimates. Do not pool repeated test images across seeds as independent observations. Compare attacks within a pair on matched indices and the same clean eligibility set. Cross-pair comparisons need a common eligibility subset as a sensitivity analysis. Wilson intervals in the built-in report describe fixed-checkpoint binomial uncertainty, not uncertainty over model training. For the full finite test set, these are descriptive intervals under a broader sampling interpretation.

## Submission gates still open

- Named venue, official template, related-work update targeted to that venue, page limit, and anonymization.
- Author confirmation of contact information and desired license. The current affiliation is Independent Researcher.
- Any venue-required statement on writing or coding assistance. This revision was prepared with AI assistance; the author must verify its scientific content.
- A clean release commit or tag plus an archival location for checkpoints, per-example logs, manifests, and generated tables. The completed manifests record source hashes but also `git_dirty: true`.
- A final author check of the compiled manuscript against the generated study summary.

The experiments support a focused workshop paper about conditioning, transfer direction, and budget sensitivity. Acceptance cannot be guaranteed.
