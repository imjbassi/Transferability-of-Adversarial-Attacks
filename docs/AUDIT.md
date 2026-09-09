# Audit of the original artifact

Audited commit: `58f8102b128b89ee3357ff56f9cf634eac5f6bcd`.

| Finding | Original location | Consequence | Correction |
|---|---|---|---|
| ImageNet weights and 1,000 output classes tested against CIFAR-10 labels | FGSM_transfer.py, TABLE1and2.py, TABLE3.py, FIG1.py | Class indices do not represent the same task | Train ten-class CIFAR-10 models and validate checkpoint metadata |
| No released CIFAR training or compatible checkpoints | Entire artifact | Approximately 90% accuracy assertion cannot be verified | Add training, validation selection, and provenance |
| Unconditional target error labeled transfer success | All table scripts | Pre-existing mistakes inflate apparent attack effectiveness | Report clean accuracy, pairwise and source-success-conditioned metrics |
| `rel_stepsize=0.01` differs from stated absolute `2/255` | TABLE1and2.py | Attack recipe differs from paper | Explicit absolute step-size CLI |
| Ten CW steps and ambiguous convergence claim | TABLE1and2.py | Optimization strength not demonstrated | Explicit CW parameters and required convergence study |
| L2 and L-infinity budgets not distinguished adequately | TABLE1and2.py | Attack comparison conflates threat models | Separate CW-L2 evaluation |
| Unseeded subset and inconsistent sample counts | TABLE1and2.py and manuscript | Results not reconstructable | Save exact seeded test indices |
| Different normalization for visualization | FIG1.py | Figure budget differs from table budget | Render actual saved evaluation inputs |
| No prediction logs or confidence intervals | Entire artifact | Counts and uncertainty cannot be audited | CSV predictions, summary counts, Wilson intervals |
| Missing license referenced in README | README.md | Reuse terms unclear | Remove unsupported license assertion without inventing terms |

The audit establishes what the released code does. It does not establish whether separate, unavailable experiments were conducted. Historical numbers are excluded from the revised evidence rather than replaced with guessed values. Original documents and images are archived for transparency. Original executable files remain available in Git history.
