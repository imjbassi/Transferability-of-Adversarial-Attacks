# Submission readiness review

The corrected manuscript is suitable for author critique as a methodological draft. It is not yet a submission-ready empirical paper. Pilot execution does not close the experimental gates below. No pilot measurements have been inserted into the manuscript.

## What is defensible now

- The audit identifies concrete defects in the released original code without speculating about unavailable experiments.
- The label space, differentiable preprocessing, pixel budgets, clean eligibility sets, source-only attack selection, and undefined denominators are explicit.
- The algebra explains why unconditional error is not transfer success. The manuscript distinguishes diagonal controls from off-diagonal transfer and distinguishes fixed-checkpoint intervals from training variability.

## Highest-priority critique

1. **Contribution:** correcting this repository and applying established attacks to three CNNs is a narrow contribution. Decide whether to pursue a correction/reproducibility paper or formulate a specific empirical question before launching the expensive study. A broad claim about architecture diversity is not established by three architectures with different capacities and optimization behavior.
2. **Evidence:** execute the frozen full-data training recipe for all nine checkpoints, inspect validation learning curves, and document any validation-only tuning. Then run matched full-test evaluations and the predeclared sensitivities in EXPERIMENTS.md. Do not choose models or attack settings because they produce appealing test transfer rates.
3. **Attack adequacy:** five-step pilot PGD and five-step CW only exercise execution. Use the planned primary settings, zero-budget controls, and step/restart/radius sensitivity. Low white-box ASR may indicate a weak attack; high conditional transfer with a tiny denominator may be uninformative. Include denominators and checkpoint-specific clean performance throughout.
4. **Inference:** present seeds individually and quantify variation across training runs. The current report does not implement paired resampling, cross-seed aggregation, or the all-model jointly correct sensitivity subset. Implement those analyses before claiming statistical differences. Do not pool repeated images as independent observations.
5. **Positioning:** the six current references support foundational definitions, but do not constitute a contemporary transfer-evaluation literature review. Update related work after selecting the actual question and workshop; distinguish this contribution from existing evaluation guidance and transfer benchmarks.
6. **Artifact:** retain full manifests, prediction CSVs, checkpoint hashes, training histories, exact software environment, and runnable commands. Large checkpoints/data should be released through an appropriate artifact channel rather than committed into Git. A saved example is an illustration, not evidence of perceptual invisibility.

## Author decisions required before submission

- Name the workshop and track; check its current scope, deadline, official template, page limits, anonymity, and artifact policy.
- Confirm affiliation/contact details and authorship. Choose repository licensing terms; no license has been invented on the author's behalf.
- Adapt the manuscript to the official template, compile and inspect it, and apply the venue's assistance-disclosure requirements.
- Replace evidence-status prose only when the corresponding final experiment artifacts exist and have been audited.

The manuscript and its existing PDF were not changed in this pass. The review above concerns the supplied draft; no current venue eligibility or acceptance claim has been verified.
