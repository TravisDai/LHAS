All values: predicted time per training iteration in ms (analytical model; hypothetical P100 PCIe reference; approved optical parameters; B = 1024; raw-input gradient retained; optimizer state 0). '—' = not run or not completed; 'infeasible' = violates the 12 GiB modeled budget.

| workload | N | LHAS | DP (p=N) | DP best (p) | OWT (adapted) | FlexFlow-MCMC (adapted) | GPipe (proposed) | single node | LHAS/DP(N) speedup | search mode |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet | 64 | 65.7 | 239.0 | 202.4 (8) | 101.3 | 71.4 | 123.2 | 943.7 | 3.64× | cost-only minimizer is memory-feasible (equals lower bound) |
| alexnet | 128 | 66.8 | 276.0 | 202.4 (8) | 133.2 | 71.5 | 206.4 | 943.7 | 4.13× | cost-only minimizer is memory-feasible (equals lower bound) |
| alexnet | 256 | 73.1 | 298.0 | 202.4 (8) | 216.8 | 78.7 | 248.5 | 943.7 | 4.08× | cost-only minimizer is memory-feasible (equals lower bound) |
| alexnet | 512 | 73.1 | 500.0 | 202.4 (8) | 571.6 | 87.7 | 283.6 | 943.7 | 6.84× | cost-only minimizer is memory-feasible (equals lower bound) |
| alexnet | 1024 | 73.1 | 587.0 | 202.4 (8) | 1635.6 | 93.3 | 332.6 | 943.7 | 8.03× | cost-only minimizer is memory-feasible (equals lower bound) |
| vgg16 | 64 | 456.5 | 874.0 | 874.0 (64) | 518.7 | 456.9 | 774.9 | 20440.7 (memory-infeasible) | 1.91× | cost-only minimizer is memory-feasible (equals lower bound) |
| vgg16 | 128 | 323.5 | 760.5 | 760.5 (128) | 453.0 | 336.4 | 824.6 | 20440.7 (memory-infeasible) | 2.35× | cost-only minimizer is memory-feasible (equals lower bound) |
| vgg16 | 256 | 283.0 | 730.7 | 730.7 (256) | 580.7 | 303.5 | 684.1 | 20440.7 (memory-infeasible) | 2.58× | cost-only minimizer is memory-feasible (equals lower bound) |
| vgg16 | 512 | 353.0 | 1129.7 | 787.4 (128) | 1402.8 | 378.8 | 663.8 | 20440.7 (memory-infeasible) | 3.20× | cost-only minimizer is memory-feasible (equals lower bound) |
| googlenet | 64 | 84.0 | 87.5 | 87.5 (64) | 90.3 | 84.3 | — | 2031.9 (memory-infeasible) | 1.04× | optimal (unconstrained minimizer is memory-feasible) |
| googlenet | 128 | 95.3 | 100.0 | 99.0 (64) | 110.2 | 95.5 | — | 2031.9 (memory-infeasible) | 1.05× | optimal (unconstrained minimizer is memory-feasible) |
| googlenet | 256 | 95.4 | 143.0 | 99.0 (64) | 178.9 | 137.1 | — | 2031.9 (memory-infeasible) | 1.50× | optimal (unconstrained minimizer is memory-feasible) |
| googlenet | 512 | 95.4 | 301.9 | 99.0 (64) | 428.8 | 287.0 | — | 2031.9 (memory-infeasible) | 3.16× | optimal (unconstrained minimizer is memory-feasible) |
| resnet50 | 64 | 236.5 | 236.5 | 236.5 (64) | 293.7 | 236.5 | — | 5839.9 (memory-infeasible) | 1.00× | optimal (unconstrained minimizer is memory-feasible) |
| resnet50 | 128 | 246.5 | 246.5 | 246.5 (128) | 450.1 | 246.5 | — | 5839.9 (memory-infeasible) | 1.00× | optimal (unconstrained minimizer is memory-feasible) |
| resnet50 | 256 | 253.2 | 321.6 | 253.2 (64) | 1503.2 | 321.6 | — | 5839.9 (memory-infeasible) | 1.27× | optimal (unconstrained minimizer is memory-feasible) |
| resnet50 | 512 | 253.2 | 643.4 | 253.2 (64) | 7234.0 | 643.4 | — | 5839.9 (memory-infeasible) | 2.54× | optimal (unconstrained minimizer is memory-feasible) |

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
* googlenet N=128: `DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP1`
* googlenet N=256: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* googlenet N=512: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP1`
* resnet50 N=64: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`
* resnet50 N=128: `DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128 DP128`
* resnet50 N=256: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`
* resnet50 N=512: `DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64 DP64`

### Planner ablations

| workload | N | LHAS | strategy only (fixed counts) | DP only, per-layer counts | MP only, per-layer counts | greedy layer-wise |
|---|---|---|---|---|---|---|
| alexnet | 256 | 73.1 | 216.8 | 110.6 | 601.9 | 83.0 |
| alexnet | 64 | 65.7 | 101.3 | 107.0 | 601.9 | 65.7 |
| vgg16 | 256 | 283.0 | 580.7 | 366.4 | infeasible | 283.0 |
| vgg16 | 64 | 456.5 | 518.7 | 549.6 | infeasible | 456.5 |

### Sensitivity at N = 64 (LHAS / DP(N) / OWT, ms)

| setting | alexnet | vgg16 | googlenet | resnet50 |
|---|---|---|---|---|
| alpha_0.25 | 60.9 / 236.2 / 98.5 | 437.3 / 859.1 / 502.7 | 79.1 / 82.6 / 85.4 | 221.1 / 221.1 / 278.3 |
| alpha_0.5 | 56.2 / 233.5 / 95.7 | 417.6 / 843.7 / 486.7 | 74.1 / 77.6 / 80.4 | 205.8 / 205.8 / 262.9 |
| alpha_0.75 | 51.3 / 230.7 / 92.9 | 396.4 / 826.9 / 470.0 | 69.1 / 72.6 / 75.4 | 188.2 / 190.4 / 247.5 |
| alpha_1.0 | 45.7 / 227.8 / 90.1 | 372.4 / 809.6 / 452.6 | 64.2 / 67.7 / 70.4 | 170.2 / 174.9 / 232.0 |
| eps_0.5 | 100.9 / 429.3 / 173.9 | 557.6 / 1383.2 / 686.0 | 123.7 / 131.9 / 135.3 | 360.9 / 360.9 / 471.2 |
| family_deepest | 125.7 / 900.8 / 326.8 | 681.6 / 2320.2 / 1056.2 | 175.6 / 196.9 / 199.9 | 510.8 / 553.2 / 611.2 |
| family_one-stage | 67.8 / infeasible / 107.6 | 459.1 / infeasible / 516.2 | 96.0 / 99.7 / 102.9 | 254.6 / 254.6 / 312.2 |
| lamW_128 | 62.9 / 239.0 / 98.0 | 449.6 / 874.0 / 513.6 | 84.0 / 87.5 / 89.1 | 236.5 / 236.5 / 270.2 |
| lamW_256 | 62.0 / 239.0 / 95.9 | 447.7 / 874.0 / 510.2 | 84.0 / 87.5 / 88.3 | 236.5 / 236.5 / 257.9 |
| lamW_32 | 68.0 / 269.1 / 119.5 | 465.6 / 892.2 / 558.4 | 87.0 / 90.3 / 95.4 | 242.6 / 242.6 / 346.8 |
| nominal | 65.7 / 239.0 / 101.3 | 456.5 / 874.0 / 518.7 | 84.0 / 87.5 / 90.3 | 236.5 / 236.5 / 293.7 |
| raw_omit | 65.2 / 238.6 / 100.8 | 455.9 / 873.4 / 518.1 | 83.2 / 86.7 / 89.5 | 235.7 / 235.7 / 292.8 |
| reach_15 | 62.5 / 265.0 / 105.7 | 453.3 / 884.2 / 532.7 | 75.2 / 77.9 / 80.4 | 219.1 / 219.1 / 269.3 |
| reach_16 | 62.3 / 264.4 / 104.8 | 452.7 / 883.5 / 531.1 | 75.1 / 77.8 / 80.3 | 218.8 / 218.8 / 267.6 |
| reach_unrestricted_nopeer | 58.3 / 227.9 / 84.5 | 436.6 / not completed / 491.6 | 60.3 / 62.2 / 64.3 | 183.2 / 183.2 / 226.7 |
| redeff_0.25 | 67.5 / 272.6 / 113.8 | 466.5 / 916.1 / 547.7 | 86.0 / 89.7 / 92.5 | 252.2 / 252.2 / 312.8 |
| redeff_1.0 | 64.8 / 222.3 / 95.0 | 450.3 / 851.8 / 503.7 | 83.0 / 86.4 / 89.1 | 227.2 / 227.2 / 282.8 |
| rho_0.25 | 89.2 / 253.8 / 116.0 | 794.1 / 1193.4 / 838.1 | 115.4 / 118.5 / 121.3 | 321.0 / 321.0 / 378.1 |
| rho_0.75 | 56.3 / 234.1 / 96.4 | 342.5 / 767.6 / 412.2 | 73.3 / 77.2 / 80.0 | 208.4 / 208.4 / 265.5 |
| rho_1.0 | 51.7 / 231.7 / 93.9 | 285.5 / 714.3 / 359.0 | 67.9 / 72.1 / 74.9 | 194.3 / 194.3 / 251.4 |
| tsetup_100e-6 | 67.7 / 240.7 / 105.3 | 460.0 / 877.0 / 524.2 | 113.4 / 116.5 / 125.0 | 263.8 / 263.8 / 323.3 |
| tsetup_5e-6 | 65.1 / 238.6 / 100.2 | 455.6 / 873.2 / 517.2 | 75.9 / 79.5 / 80.8 | 229.1 / 229.1 / 285.6 |
| whole_item | 71.0 / 496.6 / 183.4 | 508.2 / 1411.5 / 723.0 | 87.9 / 94.3 / 97.2 | 292.7 / 303.1 / 426.1 |
