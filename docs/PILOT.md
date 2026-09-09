# Real CIFAR-10 pilot: execution validation only

Executed 2026-09-09 on Windows 11, Python 3.12.5, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and an NVIDIA RTX 4070. Command:

```powershell
.\.venv\Scripts\python.exe -m transferlab.pilot --output runs/pilot-20260909 --device cuda
```

The run completed in **82.3 seconds** after installation and archive download. The official archive matched torchvision's MD5 `c58f30108f718f92721af3b95e74349a`; a chunked download from the official Toronto server took 159.7 seconds. These times depend on the machine, cache, and network.

| Stage | Wall seconds |
|---|---:|
| ResNet-18 training | 14.27 |
| VGG16 training | 10.15 |
| MobileNetV2 training | 9.67 |
| Linf evaluation, 256 images | 11.85 |
| Zero-budget evaluation, 32 images | 7.85 |
| CW wiring check, 8 images | 8.70 |
| Three reports and figure | 19.81 |

Every training run used seed 0, two epochs, 2,048 training samples, 512 validation samples, and batch size 64, preserving the fixed split seed 1729. Selected validation accuracies were 11.52%, 7.42%, and 14.65%, respectively: these checkpoints are plainly undertrained. The pilot neither establishes useful clean accuracy nor supports comparisons of transfer rates. Evaluation used batch size 32, PGD with five steps/one restart, and CW with five steps/one search. These are wiring checks, not the primary attack settings.

The 36 Linf groups contain 9,216 prediction rows; the 36 zero-budget groups contain 1,152; the nine CW groups contain 72. All 10,440 rows were checked independently for metric counts, recorded indices, and norm compliance. Clean and zero-budget predictions exactly equal their clean counterparts. All three reports also verified hashes and recomputed their summaries from CSV. Source-only construction and the corrected clean eligibility sets remain unchanged.

Raw manifests, histories, prediction CSVs, reports, and example artifacts remain in the ignored local run directory; a `pilot-validation.zip` copy was delivered alongside this review, excluding the large `best.pt` checkpoints. Those checkpoints remain locally in the three architecture subdirectories and their hashes are recorded. [pilot-validation.json](pilot-validation.json) preserves machine-readable timing/provenance/checks in Git, and [pilot-environment.txt](pilot-environment.txt) records installed versions (an environment inventory, not a portable lockfile).

The source was a working tree based on `1ab8e349876a8f8630a9665bf73c03554421d3db`. Source hashes in the record refer to actual local bytes, which can differ from Git blob hashes because of Windows line endings. No source changes were made during the real-data pilot. No pilot rates or timings were added to the manuscript as final empirical findings.
