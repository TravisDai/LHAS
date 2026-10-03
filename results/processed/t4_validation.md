# Tesla T4 compute profile: coverage and constant-throughput model validation

Source: `results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip` (sha256 e1648ca8e6dbb6a1…; notebook v2); GPU Tesla T4, CC 7.5, driver 580.82.07, CUDA 12.8, cuDNN 91900, torch 2.11.0+cu128, torchvision 0.26.0+cu128; float32, TF32 off; manifest check: match. Local computation only; no optical, multi-GPU or iteration-time measurement. Protocol: `docs/t4_validation_protocol.md`.

## Coverage

* Profiled records (shape IDs): 1783; measured 1759; not measured (memory cap): 24; repeated IDs removed: 0.
* Distinct timed kernel signatures: 1542 measured (1566 over all records). 114 signatures were timed more than once (331 records), because the notebook keyed records by the parent width too.
* Layer/configuration entries: 2770 measured of 2801. Unmeasured: vgg16/features_0 (4), vgg16/features_12 (2), vgg16/features_14 (2), vgg16/features_2 (10), vgg16/features_5 (3), vgg16/features_7 (10).
* Raw-sample consistency: 21108 summary values recomputed from the raw samples, 0 mismatches.
* Repeatability, repetitions within one session on one GPU (30 after 10 warm-up iterations); not run-to-run or device-to-device variation. Median IQR/median forward 3.1% (p90 20.4%), input_grad 3.0% (p90 14.4%), weight_grad 3.7% (p90 19.6%).

## Constant-throughput model, grouped split (no shared signatures)

Seed 0: 771 calibration signatures (879 records), 771 validation signatures (880 records). Fitted rate R = 2.913 TFLOP/s, 36% of the 8.1 TFLOP/s FP32 datasheet peak.

| metric | all | CONV | FC | all, seeds 0–99: median (5th–95th pct) |
|---|---|---|---|---|
| absolute relative error, median | 59.2% | 58.7% | 62.3% | 60.8% (57.5%–63.7%) |
| absolute relative error, 90th percentile | 96.0% | 95.6% | 97.3% | 96.2% (94.8%–97.7%) |
| absolute relative error, 95th percentile | 100.6% | 101.6% | 98.1% | 100.8% (97.6%–132.1%) |
| absolute relative error, maximum | 314.8% | 314.8% | 102.2% | 298.2% (249.6%–350.9%) |
| signed relative error, median | -41.2% | -41.2% | -16.9% | -43.6% (-48.9%–-36.8%) |
| weighted absolute error Σ|pred−meas|/Σmeas | 63.2% | 64.3% | 13.0% | 64.6% (58.7%–74.0%) |
| error of the summed time (Σpred−Σmeas)/Σmeas | 0.9% | 1.1% | -7.6% | -0.9% (-17.8%–21.6%) |
| Spearman rank correlation (global) | 0.902 | 0.894 | 0.910 | 0.904 (0.894–0.914) |

The declared split (seed 0) has a median absolute relative error at the 15th percentile of the 100 splits; the other seeds only show the split sensitivity.

Fraction of validation signatures overpredicted: 24.0%. Spearman over all 1542 signatures (split-independent): 0.904. Summed errors are over kernel-signature samples, not over any network's iteration.

### Same calibration signatures, other model forms (comparison only; the planner uses the zero-intercept form)

| model | estimator | intercept (ms) | rate (TFLOP/s) | median | p90 | p95 | weighted |
|---|---|---|---|---|---|---|---|
| zero-intercept | aggregate ratio sum(3F)/sum(T) (primary) | 0.000 | 2.91 | 59.2% | 96.0% | 100.6% | 63.2% |
| zero-intercept | least squares of relative residuals | 0.000 | 3.89 | 60.3% | 95.0% | 97.6% | 52.6% |
| affine a + 3F/R' | least squares of relative residuals | 0.252 | 4.47 | 42.6% | 91.3% | 94.5% | 51.2% |

## Configuration ranking within a layer (split-independent)

Over 136 layers with at least two measured configurations, using the time the planner reads for each configuration:

* Pairs with different local FLOPs: 26393; measured order agrees with the FLOP order in 90.5% (alexnet 89.1%, googlenet 90.5%, resnet50 91.1%, vgg16 89.6%).
* Same strategy, consecutive measured counts: 2500 pairs; 91.1% measured faster at the larger count.
* DP(p) and MP(p) pairs with equal local FLOPs (the constant-throughput model cannot order them): 1066; measured slower/faster ratio median 3.05, 90th percentile 17.99; ratio above 1.25 in 75.8%; DP faster in 83.8% (893), MP faster in 43; 130 pairs are p = 1, where DP and MP time the same kernel (ratio 1). For the 936 pairs with p > 1: median ratio 3.77, 90th percentile 18.59.

Global Spearman correlation measures ordering across shapes of very different size; the within-layer figures above measure the ordering among competing configurations of one layer.

## Direct-FLOP-equivalent rate by shape size (one point per measured signature)

| 3 × FLOPs per product | signatures | median (TFLOP/s) | 10th–90th percentile | share of measured time |
|---|---|---|---|---|
| 1e+06–1e+08 | 98 | 0.18 | 0.06–0.35 | 0.000 |
| 1e+08–1e+09 | 247 | 0.80 | 0.20–1.63 | 0.004 |
| 1e+09–1e+10 | 484 | 1.92 | 0.19–4.03 | 0.056 |
| 1e+10–1e+11 | 457 | 1.86 | 0.55–5.34 | 0.227 |
| 1e+11–1e+12 | 219 | 2.99 | 1.38–6.34 | 0.396 |
| 1e+12–1e+14 | 37 | 5.14 | 3.30–8.61 | 0.317 |

70 signature products exceed the FP32 peak when their FLOPs are counted as direct convolutions (conv 3x3 stride 1: 43, conv 5x5 stride 1: 27). This is consistent with cuDNN selecting Winograd or FFT algorithms; the algorithms were not identified by measurement.

## DP scaling efficiency by local batch

Efficiency T(p0)·p0 / (T(p)·p), p0 = the layer's smallest measured DP count (1 = linear speedup). For the following layers p0 > 1, so their reference is not the full batch (the full-batch shape exceeded the notebook's memory cap): vgg16/features_0 (p0 = 4), vgg16/features_12 (p0 = 2), vgg16/features_14 (p0 = 2), vgg16/features_2 (p0 = 8), vgg16/features_5 (p0 = 2), vgg16/features_7 (p0 = 4).

| local batch | layer configurations | median | 10th–90th percentile |
|---|---|---|---|
| 1024 | 130 | 1.00 | 1.00–1.00 |
| 512 | 133 | 1.01 | 0.90–1.19 |
| 256 | 135 | 1.05 | 0.88–1.57 |
| 128 | 136 | 1.11 | 0.93–1.83 |
| 64 | 136 | 1.13 | 0.95–2.30 |
| 32 | 136 | 1.23 | 0.79–2.05 |
| 16 | 136 | 1.30 | 0.49–1.91 |
| 8 | 136 | 1.04 | 0.26–1.59 |
| 4 | 136 | 0.56 | 0.15–1.25 |
| 2 | 136 | 0.39 | 0.09–1.00 |
| 1 | 136 | 0.29 | 0.05–0.71 |

## Local reduction

median model-equivalent rate (a+2)V/t of the fused sum, V >= 16 MiB: 218.6 GB/s (12 measurements). The rate at which the planner's reduction term reproduces the measured time; not a DRAM bandwidth measurement.

## Superseded: the notebook's record-ID split (documents finding T1)

879 calibration and 880 validation records; 71 signatures on both sides, 116 validation records with a calibration twin. Recomputed from the CSV (agrees with the notebook's JSON: True): median 69.2%, p90 99.0%, p95 143.2%, maximum 414.0%, signed median -39.1%, weighted 81.0%, summed 30.2%, Spearman 0.886. Not used for any claim.

Cross-check of the reviewer's illustrative string-typed grouped split: median 60.6% on 771 validation signatures.
