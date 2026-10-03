Predicted time per training iteration (ms) from the analytical model: hypothetical P100 PCIe compute reference, approved optical parameters, B = 1024, synchronized BatchNorm, raw-input gradient omitted, SGD momentum state, 1 GiB runtime reserve inside the 12 GiB budget. † incumbent (best completed result, some DP counts unresolved); ‡ Metropolis search with unresolved proposals; n.e. = not established; infeasible = proven by exact search or reservation lower bound; — = not run.

### Main comparison (N = 64-512; AlexNet N = 1024 supplementary)

| workload | N | LHAS | DP (p=N) | DP best (p) | OWT | FlexFlow-MCMC | GPipe | LHAS vs DP(N) | LHAS vs best baseline | LHAS status |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | 64 | 65.2 | 238.6 | 198.5 (8) | 100.8 | 65.2 | 116.2 | 3.66× | 1.00× | feasible |
| alexnet | 128 | 66.5 | 275.7 | 198.5 (8) | 132.9 | 73.5 | 120.0 | 4.14× | 1.10× | feasible |
| alexnet | 256 | 72.6 | 297.9 | 198.5 (8) | 216.6 | 74.6 | 131.2 | 4.10× | 1.03× | feasible |
| alexnet | 512 | 72.6 | 499.9 | 198.5 (8) | 571.6 | 77.9 | 131.2 | 6.89× | 1.07× | feasible |
| alexnet | 1024 (suppl.) | 72.6 | 587.0 | 198.5 (8) | 1635.6 | 77.1 | 131.2 | 8.09× | 1.06× | feasible |
| vgg16 | 64 | 455.9 | 875.1 | 875.1 (64) | 519.8 | 455.9‡ | 634.7 | 1.92× | 1.00× | feasible |
| vgg16 | 128 | 323.2 | 760.2 | 760.2 (128) | 452.7 | 353.0‡ | 492.8 | 2.35× | 1.09× | feasible |
| vgg16 | 256 | 282.8 | 730.6 | 730.6 (256) | 580.6 | 306.1‡ | 566.3 | 2.58× | 1.08× | feasible |
| vgg16 | 512 | 352.8 | 1129.6 | 787.1 (128) | 1402.7 | 394.9‡ | 659.6 | 3.20× | 1.12× | feasible |
| googlenet | 64 | 83.2 | 86.7 | 86.7 (64) | 89.5 | 83.5‡ | n/a | 1.04× | 1.00× | feasible |
| googlenet | 128 | 94.6 | 99.6 | 98.2 (64) | 109.8 | 94.8‡ | n/a | 1.05× | 1.00× | feasible |
| googlenet | 256 | 94.6 | 142.8 | 98.2 (64) | 178.7 | 96.1‡ | n/a | 1.51× | 1.02× | feasible |
| googlenet | 512 | 94.6 | 301.8 | 98.2 (64) | 428.7 | 96.1‡ | n/a | 3.19× | 1.02× | feasible |
| resnet50 | 64 | 235.7 | 235.7 | 235.7 (64) | 292.8 | 235.7‡ | n/a | 1.00× | 1.00× | feasible |
| resnet50 | 128 | 246.1 | 246.1 | 246.1 (128) | 449.7 | 246.1‡ | n/a | 1.00× | 1.00× | feasible |
| resnet50 | 256 | 252.4 | 321.4 | 252.4 (64) | 1503.0 | 252.4‡ | n/a | 1.27× | 1.00× | feasible |
| resnet50 | 512 | 252.4 | 643.3 | 252.4 (64) | 7233.9 | 252.4‡ | n/a | 2.55× | 1.00× | feasible |

### GPipe (adapted, contract D13)

| workload | N | GPipe | S | R | active nodes | M | remat | degenerate DP | status | evaluated / candidates |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | 64 | 116.2 | 4 | 16 | 64 | 32 | False | False | feasible | 194 / 466 |
| alexnet | 128 | 120.0 | 5 | 16 | 80 | 32 | False | False | feasible | 285 / 556 |
| alexnet | 256 | 131.2 | 5 | 8 | 40 | 32 | False | False | feasible | 377 / 638 |
| alexnet | 512 | 131.2 | 5 | 8 | 40 | 32 | False | False | feasible | 443 / 704 |
| alexnet | 1024 | 131.2 | 5 | 8 | 40 | 32 | False | False | feasible | 493 / 754 |
| vgg16 | 64 | 634.7 | 4 | 16 | 64 | 32 | False | False | feasible | 33 / 754 |
| vgg16 | 128 | 492.8 | 7 | 16 | 112 | 32 | False | False | feasible | 95 / 940 |
| vgg16 | 256 | 566.3 | 10 | 16 | 160 | 32 | False | False | feasible | 244 / 1118 |
| vgg16 | 512 | 659.6 | 2 | 256 | 512 | 4 | False | False | feasible | 446 / 1280 |

### Planner ablations

| workload | N | LHAS | strategy only (fixed counts) | DP only, per-layer counts | MP only, per-layer counts | greedy layer-wise |
|---|---|---|---|---|---|---|
| alexnet | 64 | 65.2 | 100.8 | 106.5 | 490.1 | 65.2 |
| alexnet | 256 | 72.6 | 216.6 | 110.1 | 491.4 | 82.9 |
| vgg16 | 64 | 455.9 | 519.8 | 549.0 | infeasible | 455.9 |
| vgg16 | 256 | 282.8 | 580.6 | 366.2 | infeasible | 282.8 |

### Collective-family comparison at N = 64 (LHAS re-planned with each candidate family)

| workload | all families | one-stage | deepest |
|---|---|---|---|
| alexnet | 65.2 | 67.3 | 125.2 |
| vgg16 | 455.9 | 458.5 | 681.0 |
| googlenet | 83.2 | 95.2 | 174.0 |
| resnet50 | 235.7 | 253.8 | 510.0 |

### Batch-size sensitivity at N = 64

| workload | B | N | LHAS | DP (p=N) | DP best (p) | OWT |
|---|---|---|---|---|---|---|
| alexnet | 1024 | 64 | 65.2 | 238.6 | 198.5 (8) | 100.8 |
| alexnet | 256 | 64 | 26.8 | 227.9 | 112.9 (8) | 41.1 |
| vgg16 | 1024 | 64 | 455.9 | 875.1 | 875.1 (64) | 519.8 |
| vgg16 | 256 | 64 | 164.2 | 634.3 | 634.3 (64) | 184.8 |
| googlenet | 1024 | 64 | 83.2 | 86.7 | 86.7 (64) | 89.5 |
| googlenet | 256 | 64 | 52.8 | 63.5 | 56.0 (32) | 62.6 |
| resnet50 | 1024 | 64 | 235.7 | 235.7 | 235.7 (64) | 292.8 |
| resnet50 | 256 | 64 | 147.9 | 167.9 | 151.9 (32) | 186.8 |

### One-at-a-time sensitivity at N = 64 (LHAS / DP(N) / OWT, ms)

| setting | alexnet | vgg16 | googlenet | resnet50 |
|---|---|---|---|---|
| alpha_0.25 | 60.4 / 235.8 / 98.0 | 436.7 / 859.1 / 503.1 | 78.3 / 81.8 / 84.5 | 220.3 / 220.3 / 277.4 |
| alpha_0.5 | 55.7 / 233.0 / 95.2 | 417.0 / 843.1 / 486.3 | 73.3 / 76.8 / 79.6 | 204.9 / 204.9 / 262.0 |
| alpha_0.75 | 50.8 / 230.2 / 92.4 | 395.8 / 826.4 / 469.4 | 68.3 / 71.8 / 74.6 | 187.4 / 189.5 / 246.6 |
| alpha_1.0 | 45.2 / 227.4 / 89.6 | 371.8 / 809.0 / 452.0 | 63.4 / 66.8 / 69.6 | 169.4 / 174.1 / 231.2 |
| eps_0.5 | 100.4 / 428.8 / 173.4 | 557.0 / 1388.7 / 691.1 | 122.9 / 131.1 / 134.5 | 360.1 / 360.1 / 470.4 |
| lamW_128 | 62.4 / 238.6 / 97.6 | 449.0 / 875.1 / 513.9 | 83.2 / 86.7 / 88.3 | 235.7 / 235.7 / 269.4 |
| lamW_256 | 61.5 / 238.6 / 95.4 | 447.1 / 875.1 / 511.3 | 83.2 / 86.7 / 87.5 | 235.7 / 235.7 / 257.1 |
| lamW_32 | 67.5 / 268.7 / 119.0 | 465.0 / 891.6 / 557.8 | 86.2 / 89.5 / 94.6 | 241.7 / 241.7 / 346.0 |
| no_momentum | 65.2 / 238.6 / 100.8 | 455.9 / 874.3 / 519.8 | 83.2 / 86.7 / 89.5 | 235.7 / 235.7 / 292.8 |
| nominal | 65.2 / 238.6 / 100.8 | 455.9 / 875.1 / 519.8 | 83.2 / 86.7 / 89.5 | 235.7 / 235.7 / 292.8 |
| raw_retain | 65.7 / 239.0 / 101.3 | 456.5 / 875.7 / 520.4 | 84.0 / 87.5 / 90.3 | 236.5 / 236.5 / 293.7 |
| reach_15 | 62.0 / 264.5 / 105.2 | 452.7 / 883.6 / 532.2 | 74.4 / 77.1 / 79.6 | 218.2 / 218.2 / 268.5 |
| reach_16 | 61.8 / 263.9 / 104.3 | 452.1 / 882.9 / 530.5 | 74.3 / 77.0 / 79.5 | 218.0 / 218.0 / 266.8 |
| reach_unrestricted_nopeer | 57.8 / 228.6 / 84.0 | 436.0 / n.e. / 494.4 | 59.5 / 61.4 / 63.5 | 182.4 / 182.4 / 225.9 |
| redeff_0.25 | 67.0 / 272.1 / 113.3 | 465.9 / 915.5 / 547.1 | 85.2 / 88.9 / 91.7 | 251.4 / 251.4 / 312.0 |
| redeff_1.0 | 64.3 / 221.8 / 94.5 | 449.8 / 854.3 / 505.9 | 82.2 / 85.6 / 88.3 | 226.4 / 226.4 / 282.0 |
| reserve_0 | 65.2 / 238.6 / 100.8 | 455.9 / 874.3 / 518.9 | 83.2 / 86.7 / 89.5 | 235.7 / 235.7 / 292.8 |
| rho_0.25 | 88.3 / 252.8 / 115.1 | 792.9 / 1193.9 / 838.6 | 113.7 / 116.9 / 119.6 | 319.3 / 319.3 / 376.5 |
| rho_0.75 | 56.0 / 233.8 / 96.0 | 342.1 / 768.8 / 413.5 | 72.7 / 76.7 / 79.5 | 207.9 / 207.9 / 265.0 |
| rho_1.0 | 51.4 / 231.4 / 93.6 | 285.2 / 715.7 / 360.4 | 67.5 / 71.7 / 74.5 | 193.9 / 193.9 / 251.0 |
| tsetup_100e-6 | 67.3 / 240.2 / 104.8 | 459.4 / 878.2 / 525.4 | 112.6 / 115.7 / 124.2 | 263.0 / 263.0 / 322.5 |
| tsetup_5e-6 | 64.6 / 238.1 / 99.7 | 455.0 / 874.2 / 518.3 | 75.1 / 78.7 / 80.0 | 228.3 / 228.3 / 284.7 |
| whole_item | 70.6 / 496.1 / 182.9 | 507.6 / 1410.9 / 722.4 | 87.1 / 93.5 / 96.4 | 291.9 / 302.3 / 425.3 |

### Verification coverage of selected operations

| workload | N | selected operations | network operations | checked (actual) | checked (reduced payload) | skipped | packing re-derived |
|---|---|---|---|---|---|---|---|
| alexnet | 64 | 22 | 12 | 12 | 0 | 0 | 12 |
| alexnet | 128 | 22 | 12 | 12 | 0 | 0 | 12 |
| alexnet | 256 | 22 | 12 | 12 | 0 | 0 | 12 |
| alexnet | 512 | 22 | 12 | 12 | 0 | 0 | 12 |
| alexnet | 1024 | 22 | 12 | 12 | 0 | 0 | 12 |
| vgg16 | 64 | 46 | 20 | 20 | 0 | 0 | 20 |
| vgg16 | 128 | 46 | 20 | 20 | 0 | 0 | 20 |
| vgg16 | 256 | 46 | 20 | 20 | 0 | 0 | 20 |
| vgg16 | 512 | 46 | 24 | 24 | 0 | 0 | 24 |
| googlenet | 64 | 484 | 179 | 179 | 0 | 0 | 179 |
| googlenet | 128 | 484 | 179 | 179 | 0 | 0 | 179 |
| googlenet | 256 | 484 | 179 | 179 | 0 | 0 | 179 |
| googlenet | 512 | 484 | 179 | 179 | 0 | 0 | 179 |
| resnet50 | 64 | 304 | 160 | 160 | 0 | 0 | 160 |
| resnet50 | 128 | 304 | 160 | 160 | 0 | 0 | 160 |
| resnet50 | 256 | 304 | 160 | 160 | 0 | 0 | 160 |
| resnet50 | 512 | 304 | 160 | 160 | 0 | 0 | 160 |

### Metropolis search accounting (initialization included; review 0892134, F4)

| workload | N | evaluations (sum over seeds) | of which initialization | globally distinct plans | not established (rejected) | seconds (sum over seeds) |
|---|---|---|---|---|---|---|
| alexnet | 64 | 2657 | 12 | 2624 | 0 | 4 |
| alexnet | 128 | 2726 | 12 | 2717 | 0 | 5 |
| alexnet | 256 | 2765 | 12 | 2758 | 0 | 6 |
| alexnet | 512 | 2774 | 12 | 2767 | 0 | 7 |
| alexnet | 1024 | 2788 | 12 | 2782 | 0 | 11 |
| vgg16 | 64 | 2671 | 156 | 2663 | 210 | 2075 |
| vgg16 | 128 | 2792 | 156 | 2788 | 175 | 4021 |
| vgg16 | 256 | 2825 | 156 | 2821 | 251 | 5546 |
| vgg16 | 512 | 2864 | 159 | 2858 | 205 | 1910 |
| googlenet | 64 | 2923 | 156 | 2914 | 298 | 60 |
| googlenet | 128 | 2928 | 159 | 2916 | 283 | 67 |
| googlenet | 256 | 2933 | 159 | 2923 | 280 | 73 |
| googlenet | 512 | 2982 | 159 | 2967 | 267 | 103 |
| resnet50 | 64 | 2890 | 156 | 2885 | 494 | 43 |
| resnet50 | 128 | 2941 | 156 | 2936 | 496 | 45 |
| resnet50 | 256 | 2949 | 159 | 2942 | 526 | 56 |
| resnet50 | 512 | 2979 | 159 | 2973 | 505 | 79 |

### Measured T4 computation (sensitivity study; analytical values in parentheses)

| workload | N | LHAS (ms) | DP (p=N) | DP best | OWT | DP best certification | unavailable DP counts |
|---|---|---|---|---|---|---|---|
| AlexNet | 64 | 68.5 (65.2) | 3.79 (3.66) | 2.74 (3.05) | 1.55 (1.55) | complete enumeration |  |
| AlexNet | 256 | 76.1 (72.6) | 4.21 (4.10) | 2.47 (2.73) | 2.98 (2.98) | complete enumeration |  |
| VGG16 | 64 | 412.6 (455.9) | 2.06 (1.92) | 2.06 (1.92) | 1.16 (1.14) | lower-bound exclusion | 1 2 4 |
| VGG16 | 256 | 309.5 (282.8) | 2.53 (2.58) | 2.53 (2.58) | 1.96 (2.05) | lower-bound exclusion | 1 2 4 |
| GoogLeNet | 64 | 99.3 (83.2) | 1.04 (1.04) | 1.04 (1.04) | 1.07 (1.08) | complete enumeration |  |
| GoogLeNet | 256 | 111.3 (94.6) | 1.45 (1.51) | 1.04 (1.04) | 1.77 (1.89) | complete enumeration |  |
| ResNet50 | 64 | 277.8 (235.7) | 1.00 (1.00) | 1.00 (1.00) | 1.21 (1.24) | complete enumeration |  |
| ResNet50 | 256 | 298.0 (252.4) | 1.17 (1.27) | 1.00 (1.00) | 5.14 (5.95) | complete enumeration |  |

Selected LHAS configurations:

* alexnet N=64: `DP64 DP64 DP64 DP64 DP64 MP8 MP8 DP1`
* alexnet N=128: `DP128 DP128 DP128 DP128 DP128 MP8 MP8 DP1`
* alexnet N=256: `DP64 DP64 DP64 DP64 DP64 MP8 MP8 DP1`
* alexnet N=512: `DP64 DP64 DP64 DP64 DP64 MP8 MP8 DP1`
* alexnet N=1024: `DP64 DP64 DP64 DP64 DP64 MP8 MP8 DP1`
* vgg16 N=64: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 MP8 MP8 DP1`
* vgg16 N=128: `DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 MP8 MP8 DP1`
* vgg16 N=256: `DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 MP8 MP8 DP1`
* vgg16 N=512: `DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP256 DP128 DP128 DP64 MP8 MP8 DP1`
* googlenet N=64: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* googlenet N=128: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* googlenet N=256: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* googlenet N=512: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* resnet50 N=64: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`
* resnet50 N=128: `DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128`
* resnet50 N=256: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`
* resnet50 N=512: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`
