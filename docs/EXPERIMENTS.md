# Experiment specification and remaining submission work

## Primary study

1. Train ResNet-18, adapted VGG16, and MobileNetV2 for each seed in `{0,1,2}`. Keep split seed 1729 constant. Record validation-selected checkpoints, training histories, hardware, elapsed time, and versions.
2. Freeze training choices before examining test attacks. If the recipe undertrains an architecture, tune on validation data, document the change, and rerun its planned seeds. Do not assert a target clean accuracy in advance.
3. Evaluate all 10,000 test examples using identical indices and evaluation seed 0 for each checkpoint trio. Include all nine source-target entries. The diagonal measures source attack strength; six off-diagonal entries measure transfer.
4. Run identity and random-noise controls plus FGSM and PGD at L-infinity budgets 0, 2/255, 4/255, and 8/255. Record explicit step sizes. Run PGD with 10, 40, and 100 steps and 1 and 5 restarts to examine sensitivity. Primary PGD uses 40 steps and 5 restarts at 8/255.
5. Run CW-L2 separately at radius 1.0, 1,000 steps, and nine binary-search steps, then assess step and radius sensitivity. A limited iteration count cannot establish a minimum-norm solution.
6. Generate tables using `transferlab.report`. Show source ASR, pair transfer, conditional transfer, both clean accuracies, raw counts, and norm diagnostics. Preserve prediction CSVs and manifests alongside final tables.

## Statistical interpretation

Show each training seed separately and summarize across seed-level estimates. Do not pool repeated test images across seeds as independent observations. Compare attacks within a pair on matched indices and the same clean eligibility set. Cross-pair comparisons need a common eligibility subset as a sensitivity analysis. Wilson intervals in the built-in report describe fixed-checkpoint binomial uncertainty, not uncertainty over model training. For the full finite test set, these are descriptive intervals under a broader sampling interpretation.

## Submission gates still open

- Actual trained CIFAR-10 checkpoints and completed primary/sensitivity experiments.
- Independently recomputable empirical tables and figures, tied to manifests and hashes.
- A defensible contribution beyond running established attacks on three established CNNs. Options include a narrowly framed reproducibility/correction workshop paper or an expanded empirical question supported by new experiments.
- Named venue, official template, related-work update targeted to that venue, page limit, and anonymization.
- Author confirmation of current affiliation, contact information, and desired license.
- Any venue-required statement on writing or coding assistance. This revision was prepared with AI assistance; the author must verify its scientific content.

The current manuscript is a complete methodological draft with explicit limitations. It is not a completed empirical transfer study and cannot guarantee acceptance.
