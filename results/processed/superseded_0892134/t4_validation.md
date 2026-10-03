# Tesla T4 compute profile: coverage and model validation

Source: `results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip` (notebook v2); GPU Tesla T4, CC 7.5, driver 580.82.07, CUDA 12.8, cuDNN 91900, torch 2.11.0+cu128, torchvision 0.26.0+cu128; float32, TF32 off; manifest check: match. Local computation only; no optical measurement.

## Coverage

* Unique local shapes: 1783; measured 1759 (98.7%); not measured (memory cap): 24.
* Layer/configuration entries: 2770 measured of 2801.
* Unmeasured entries (all VGG16 early CONV layers at large local batch or full-batch MP): vgg16/features_0 (4), vgg16/features_12 (2), vgg16/features_14 (2), vgg16/features_2 (10), vgg16/features_5 (3), vgg16/features_7 (10).
* Timing repeatability: median IQR/median of the forward product 0.031.
* Total measured time (sum of medians over shapes): 71.0 s.

## Constant-throughput model on held-out shapes (notebook split, seed 0)

Fitted rate R = 2.351 TFLOP/s (29% of the 8.1 TFLOP/s FP32 peak); 879 calibration and 880 validation shapes.

| metric | all | CONV | FC |
|---|---|---|---|
| signed relative error, median | -0.391 | -0.404 | -0.145 |
| absolute relative error, median | 0.692 | 0.690 | 0.719 |
| absolute relative error, 90th percentile | 0.990 | 0.984 | 1.051 |
| absolute relative error, 95th percentile | 1.432 | 1.498 | 1.146 |
| absolute relative error, maximum | 4.140 | 4.140 | 1.506 |
| Spearman rank correlation | 0.886 | 0.883 | 0.829 |

Additional (computed here from the same split): time-weighted absolute relative error sum|pred − meas| / sum(meas) = 0.810; error of the summed validation time (pred − meas)/meas = +0.302.

For comparison only, an affine model t = a + F/R' fitted on the same calibration shapes (a = 0.253 ms per layer shape, R' = 4.33 TFLOP/s) gives a median absolute relative error of 0.505 (90th percentile 0.928; time-weighted 0.504). The remaining spread reflects shape-dependent kernel efficiency.

## Achieved rate by shape size (direct-convolution FLOP count, all measured shapes)

| 3 × FLOPs per product | shapes | median rate (TFLOP/s) | 10th–90th percentile | share of measured time |
|---|---|---|---|---|
| 1e+06–1e+08 | 98 | 0.18 | 0.06–0.35 | 0.000 |
| 1e+08–1e+09 | 280 | 0.67 | 0.10–1.59 | 0.006 |
| 1e+09–1e+10 | 597 | 1.36 | 0.17–3.73 | 0.073 |
| 1e+10–1e+11 | 517 | 1.69 | 0.52–5.07 | 0.243 |
| 1e+11–1e+12 | 230 | 2.92 | 1.29–6.06 | 0.385 |
| 1e+12–1e+14 | 37 | 5.14 | 3.30–8.61 | 0.292 |

70 individual products exceed the FP32 peak when counted as direct convolutions, all stride-1 3×3 or 5×5 convolutions ([3, 3]: 43, [5, 5]: 27); this is consistent with cuDNN selecting Winograd or FFT algorithms, so direct-convolution FLOP counts overstate the arithmetic actually performed for these shapes.

## DP scaling efficiency by local batch

Efficiency = T(p_min)·p_min / (T(p)·p) for each layer's DP configurations (1 = linear speedup).

| local batch | layer configurations | median efficiency | 10th–90th percentile |
|---|---|---|---|
| 1024 | 130 | 1.00 | 1.00–1.00 |
| 512 | 133 | 1.01 | 0.90–1.19 |
| 256 | 135 | 1.03 | 0.88–1.57 |
| 128 | 136 | 1.11 | 0.93–1.83 |
| 64 | 136 | 1.13 | 0.95–2.30 |
| 32 | 136 | 1.23 | 0.79–2.05 |
| 16 | 136 | 1.30 | 0.49–1.91 |
| 8 | 136 | 1.04 | 0.26–1.59 |
| 4 | 136 | 0.56 | 0.15–1.25 |
| 2 | 136 | 0.39 | 0.09–1.00 |
| 1 | 136 | 0.29 | 0.05–0.71 |

## Local reduction

Median model-equivalent rate (a+2)V/t of the fused sum for V ≥ 16 MiB: 218.6 GB/s (12 measurements). This is the rate at which the planner's reduction term reproduces the measured time, not a DRAM bandwidth measurement; it is used as β_red in the profiled planning sensitivity.
