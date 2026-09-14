# Three-seed CIFAR-10 study

Rates are percentages. Across-seed summaries treat the checkpoint seed as the unit; they do not pool repeated test images.

## Clean accuracy

| Model | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 95.11 | 95.01 | 95.20 | 95.11 | 0.10 |
| VGG16 | 90.59 | 90.41 | 90.88 | 90.63 | 0.24 |
| MobileNetV2 | 89.97 | 90.57 | 90.38 | 90.31 | 0.31 |

## Pair transfer by budget

### $L_\infty=2/255$

| Attack | Source | Target | Mean PTR | Seed range | Mean CTR | Mean source ASR |
|---|---|---|---:|---:|---:|---:|
| NOISE | ResNet-18 | VGG16 | 0.27 | 0.24--0.33 | 2.95 | 0.40 |
| NOISE | ResNet-18 | MobileNetV2 | 0.61 | 0.54--0.66 | 5.77 | 0.40 |
| NOISE | VGG16 | ResNet-18 | 0.26 | 0.16--0.33 | 3.50 | 0.26 |
| NOISE | VGG16 | MobileNetV2 | 0.56 | 0.42--0.71 | 7.22 | 0.26 |
| NOISE | MobileNetV2 | ResNet-18 | 0.19 | 0.16--0.23 | 1.89 | 0.76 |
| NOISE | MobileNetV2 | VGG16 | 0.21 | 0.19--0.23 | 2.31 | 0.76 |
| FGSM | ResNet-18 | VGG16 | 6.64 | 6.26--7.23 | 17.39 | 39.47 |
| FGSM | ResNet-18 | MobileNetV2 | 15.76 | 15.64--15.92 | 38.75 | 39.47 |
| FGSM | VGG16 | ResNet-18 | 11.09 | 10.68--11.65 | 20.75 | 52.00 |
| FGSM | VGG16 | MobileNetV2 | 18.22 | 18.00--18.65 | 33.30 | 52.00 |
| FGSM | MobileNetV2 | ResNet-18 | 8.49 | 8.22--8.80 | 10.62 | 79.98 |
| FGSM | MobileNetV2 | VGG16 | 4.94 | 4.79--5.18 | 6.23 | 79.98 |
| PGD | ResNet-18 | VGG16 | 5.84 | 5.44--6.43 | 6.21 | 94.47 |
| PGD | ResNet-18 | MobileNetV2 | 21.61 | 21.54--21.67 | 22.96 | 94.47 |
| PGD | VGG16 | ResNet-18 | 15.45 | 15.06--15.88 | 21.42 | 71.73 |
| PGD | VGG16 | MobileNetV2 | 23.29 | 22.86--23.53 | 31.97 | 71.73 |
| PGD | MobileNetV2 | ResNet-18 | 10.79 | 10.06--11.18 | 10.81 | 99.85 |
| PGD | MobileNetV2 | VGG16 | 4.61 | 4.39--4.81 | 4.62 | 99.85 |
### $L_\infty=4/255$

| Attack | Source | Target | Mean PTR | Seed range | Mean CTR | Mean source ASR |
|---|---|---|---:|---:|---:|---:|
| NOISE | ResNet-18 | VGG16 | 0.66 | 0.56--0.73 | 7.94 | 1.05 |
| NOISE | ResNet-18 | MobileNetV2 | 1.78 | 1.59--2.04 | 22.62 | 1.05 |
| NOISE | VGG16 | ResNet-18 | 0.69 | 0.48--0.80 | 9.36 | 0.68 |
| NOISE | VGG16 | MobileNetV2 | 1.67 | 1.49--1.89 | 8.88 | 0.68 |
| NOISE | MobileNetV2 | ResNet-18 | 0.64 | 0.61--0.69 | 6.79 | 2.24 |
| NOISE | MobileNetV2 | VGG16 | 0.51 | 0.43--0.61 | 6.49 | 2.24 |
| FGSM | ResNet-18 | VGG16 | 14.17 | 13.39--15.20 | 29.40 | 48.71 |
| FGSM | ResNet-18 | MobileNetV2 | 31.25 | 30.20--31.88 | 58.16 | 48.71 |
| FGSM | VGG16 | ResNet-18 | 30.28 | 29.55--30.98 | 40.32 | 74.64 |
| FGSM | VGG16 | MobileNetV2 | 40.52 | 40.40--40.61 | 53.05 | 74.64 |
| FGSM | MobileNetV2 | ResNet-18 | 22.20 | 21.39--22.75 | 24.33 | 90.76 |
| FGSM | MobileNetV2 | VGG16 | 12.31 | 12.16--12.41 | 13.55 | 90.76 |
| PGD | ResNet-18 | VGG16 | 9.65 | 8.75--10.42 | 9.66 | 99.91 |
| PGD | ResNet-18 | MobileNetV2 | 42.40 | 41.39--43.09 | 42.44 | 99.91 |
| PGD | VGG16 | ResNet-18 | 54.52 | 53.95--54.81 | 55.69 | 97.93 |
| PGD | VGG16 | MobileNetV2 | 64.50 | 63.33--65.65 | 65.81 | 97.93 |
| PGD | MobileNetV2 | ResNet-18 | 31.80 | 28.75--33.83 | 31.80 | 100.00 |
| PGD | MobileNetV2 | VGG16 | 10.69 | 10.57--10.90 | 10.69 | 100.00 |
### $L_\infty=8/255$

| Attack | Source | Target | Mean PTR | Seed range | Mean CTR | Mean source ASR |
|---|---|---|---:|---:|---:|---:|
| NOISE | ResNet-18 | VGG16 | 1.73 | 1.57--1.94 | 16.80 | 3.55 |
| NOISE | ResNet-18 | MobileNetV2 | 7.89 | 7.34--8.93 | 51.00 | 3.55 |
| NOISE | VGG16 | ResNet-18 | 2.51 | 2.28--2.66 | 25.03 | 1.89 |
| NOISE | VGG16 | MobileNetV2 | 7.04 | 6.72--7.58 | 41.44 | 1.89 |
| NOISE | MobileNetV2 | ResNet-18 | 2.48 | 2.28--2.65 | 16.27 | 8.37 |
| NOISE | MobileNetV2 | VGG16 | 1.45 | 1.25--1.81 | 9.08 | 8.37 |
| FGSM | ResNet-18 | VGG16 | 27.80 | 26.21--30.01 | 45.50 | 58.54 |
| FGSM | ResNet-18 | MobileNetV2 | 58.30 | 56.73--59.43 | 78.15 | 58.54 |
| FGSM | VGG16 | ResNet-18 | 55.27 | 54.14--56.75 | 62.69 | 87.31 |
| FGSM | VGG16 | MobileNetV2 | 64.88 | 64.38--65.71 | 72.25 | 87.31 |
| FGSM | MobileNetV2 | ResNet-18 | 47.37 | 47.20--47.55 | 50.68 | 92.03 |
| FGSM | MobileNetV2 | VGG16 | 29.34 | 28.09--30.55 | 31.36 | 92.03 |
| PGD | ResNet-18 | VGG16 | 18.09 | 16.73--19.38 | 18.09 | 100.00 |
| PGD | ResNet-18 | MobileNetV2 | 72.27 | 71.76--72.73 | 72.27 | 100.00 |
| PGD | VGG16 | ResNet-18 | 92.73 | 92.40--92.99 | 92.75 | 99.98 |
| PGD | VGG16 | MobileNetV2 | 95.23 | 94.80--95.61 | 95.25 | 99.98 |
| PGD | MobileNetV2 | ResNet-18 | 67.60 | 63.46--70.61 | 67.60 | 100.00 |
| PGD | MobileNetV2 | VGG16 | 25.94 | 25.73--26.29 | 25.94 | 100.00 |

## Convergence checks

The largest absolute off-diagonal PTR change from 40 steps/five restarts to 100 steps/five restarts at 8/255 is 1.70 percentage points.

For the seed-0 check at 2/255, the corresponding maximum is 0.27 percentage points.

Exact seed-level numerators and denominators are in `seed_level.csv`.

Supplementary seed-specific Wilson intervals for fixed checkpoints are in `primary_pgd_wilson_envelope.csv`; the main paper reports seed ranges.

The exact conditioning identity and failed-source stratum are in `conditioning_decomposition.csv`.

## Matched learning-rate sensitivity

Seed-0 PGD PTR at 8/255 before and after training every architecture with initial learning rate 0.01:

| Source | Target | Primary | Matched LR | Difference |
|---|---|---:|---:|---:|
| ResNet-18 | VGG16 | 19.38 | 32.71 | +13.33 |
| ResNet-18 | MobileNetV2 | 71.76 | 61.66 | -10.10 |
| VGG16 | ResNet-18 | 92.99 | 95.98 | +2.99 |
| VGG16 | MobileNetV2 | 94.80 | 93.12 | -1.68 |
| MobileNetV2 | ResNet-18 | 63.46 | 37.53 | -25.93 |
| MobileNetV2 | VGG16 | 25.73 | 18.55 | -7.18 |

## Provenance note

The manifests preserve checkpoint hashes, prediction checksums, source-file hashes, and environment details. They also record a dirty working tree; archive the manifests and source snapshot with the submission.
