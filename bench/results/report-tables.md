# Second-detector benchmark

673 scored images: pilot 450 (9 arms x 50), held-out new 223 (75 DIV2K real, 74 FLUX, 74 SDXL).

## AUC on held-out set (95% bootstrap CI)

| detector | all new | real vs FLUX | real vs SDXL |
|---|---|---|---|
| swinv2 | 0.919 [0.88-0.95] | 0.895 [0.84-0.94] | 0.943 [0.90-0.97] |
| commfor | 0.925 [0.89-0.95] | 0.853 [0.79-0.91] | 0.996 [0.99-1.00] |
| siglip | 0.873 [0.83-0.92] | 0.906 [0.86-0.95] | 0.839 [0.77-0.90] |
| swin+commfor mean | 0.956 [0.93-0.98] | 0.916 [0.86-0.96] | 0.997 [0.99-1.00] |
| swin+commfor min | 0.950 [0.92-0.97] | 0.909 [0.86-0.96] | 0.991 [0.98-1.00] |
| all3 mean | 0.966 [0.94-0.99] | 0.940 [0.90-0.98] | 0.992 [0.98-1.00] |

## AUC per pilot degradation arm

| detector | A0 | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|---|---|---|
| swinv2 | 0.883 | 0.899 | 0.878 | 0.880 | 0.894 | 0.891 | 0.883 | 0.848 | 0.882 |
| commfor | 0.859 | 0.856 | 0.802 | 0.670 | 0.866 | 0.870 | 0.853 | 0.851 | 0.770 |
| siglip | 0.930 | 0.931 | 0.931 | 0.930 | 0.931 | 0.932 | 0.933 | 0.930 | 0.930 |
| swin+commfor mean | 0.906 | 0.914 | 0.910 | 0.880 | 0.910 | 0.922 | 0.901 | 0.901 | 0.899 |
| swin+commfor min | 0.878 | 0.872 | 0.843 | 0.760 | 0.886 | 0.901 | 0.867 | 0.900 | 0.824 |
| all3 mean | 0.920 | 0.922 | 0.917 | 0.917 | 0.920 | 0.936 | 0.922 | 0.930 | 0.920 |

## "Likely AI" band: false accusations vs catches

Cut-off per detector = lowest value giving <=4% real FPR on the **pilot** (all arms); then applied unchanged to the held-out set.

| detector | tuned cut-off | pilot FPR | pilot TPR | **held-out FPR** | **held-out TPR** | TPR FLUX | TPR SDXL |
|---|---|---|---|---|---|---|---|
| swinv2 | 0.9899 | 0.04 | 0.61 | 0.07 | 0.66 | 0.59 | 0.72 |
| commfor | 0.0976 | 0.04 | 0.36 | 0.12 | 0.81 | 0.64 | 0.99 |
| siglip | 0.9900 | 0.04 | 0.68 | 0.03 | 0.51 | 0.61 | 0.42 |
| swin+commfor mean | 0.6437 | 0.04 | 0.49 | 0.00 | 0.80 | 0.66 | 0.93 |
| swin+commfor min | 0.0456 | 0.04 | 0.37 | 0.01 | 0.80 | 0.69 | 0.92 |
| all3 mean | 0.2001 | 0.04 | 0.73 | 0.03 | 0.84 | 0.74 | 0.93 |

## Current production bands (Swinv2 alone, 0.99 / 0.10) on held-out set

- real: likely AI 0.07 · inconclusive 0.12 · no strong signal 0.81
- fake: likely AI 0.66 · inconclusive 0.20 · no strong signal 0.14

## Score correlation on held-out set (Spearman)

- swinv2 vs commfor: 0.62
- swinv2 vs siglip: 0.60
- commfor vs siglip: 0.44

## Paired bootstrap: AUC gain over Swinv2 on held-out set

Resamples images jointly, so both detectors see the same bootstrap sample.

| detector | ΔAUC | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| commfor | +0.006 | [-0.039, +0.051] | 0.389 |
| siglip | -0.046 | [-0.100, +0.009] | 0.952 |
| swin+commfor mean | +0.037 | [+0.010, +0.068] | 0.004 |
| swin+commfor min | +0.031 | [+0.001, +0.065] | 0.022 |
| all3 mean | +0.047 | [+0.018, +0.080] | 0.001 |

## Trivial baseline: normalized file size alone

Every image is JPEG Q95 at 1024², so size measures compressibility (content entropy), not provenance. Smaller file → predicted fake.

- held-out: AUC 0.811 [0.75-0.87] (Swinv2 0.919, swin+commfor mean 0.956)
- real vs flux: AUC 0.820
- real vs sdxl: AUC 0.803

## Operating points with 95% Wilson CIs (held-out set)

| detector @ cut-off | FPR (reals flagged) | TPR (fakes caught) |
|---|---|---|
| swinv2 @ 0.9900 | 5/75 = 0.067 [0.029-0.147] | 97/148 = 0.655 [0.576-0.727] |
| swin+commfor mean @ 0.6437 | 0/75 = 0.000 [0.000-0.049] | 118/148 = 0.797 [0.725-0.854] |

## Operating points per degradation arm (pilot, same cut-offs)

| arm | Swinv2@0.99 FPR | Swinv2 TPR | ens FPR | ens TPR |
|---|---|---|---|---|
| A0 | 0.04 | 0.68 | 0.04 | 0.56 |
| A1 | 0.00 | 0.68 | 0.04 | 0.56 |
| A2 | 0.00 | 0.52 | 0.00 | 0.36 |
| A3 | 0.00 | 0.36 | 0.00 | 0.24 |
| A4 | 0.00 | 0.68 | 0.04 | 0.56 |
| A5 | 0.00 | 0.56 | 0.00 | 0.48 |
| A6 | 0.08 | 0.68 | 0.08 | 0.56 |
| A7 | 0.24 | 0.76 | 0.16 | 0.76 |
| A8 | 0.00 | 0.52 | 0.00 | 0.32 |

## Score movement under degradation (mean |Δ| vs A0, pilot)

A detector that reads generation artefacts should move when artefacts are destroyed.

| detector | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|---|---|
| swinv2 | 0.012 | 0.114 | 0.170 | 0.018 | 0.092 | 0.009 | 0.162 | 0.143 |
| commfor | 0.029 | 0.106 | 0.140 | 0.050 | 0.089 | 0.016 | 0.089 | 0.119 |
| siglip | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
