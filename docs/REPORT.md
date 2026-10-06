# Report: final specification (D1–D17), corrections, regenerated results

Public artifact report, prepared from the development report of 2026-09-28 (revised
2026-09-29 and 2026-09-30). Every time below is a prediction of the analytical model
(hypothetical P100 PCIe reference, approved optical parameters, B = 1024, synchronized
BatchNorm, raw-input gradient omitted, SGD momentum state, 1 GiB runtime reserve inside
12 GiB). None is a measurement. The Colab notebook was run by the author on a Tesla T4 on
2026-09-28 with notebook version 2 (Section 8); we did not run it. Material that concerns
only the unpublished manuscript has been left out of this public version.

## 0. Audit history

The implementation was audited independently three times after the final specification.
The audit reports are not included; the findings, their fixes and their regression tests are
listed in `docs/change_log.md`.

* **Audit of commit 0892134.** Implementation findings F1, F3, F4, F5, T5, T6 fixed with
  regressions (`tests/test_review_0892134.py`, 11 tests, 10 of which failed before the
  fixes). T4 validation redone by timed kernel signature (T1, T3, T4; protocol
  `docs/t4_validation_protocol.md`); superseded outputs in
  `results/processed/superseded_0892134/`; no GPU rerun. All 16 primary, the
  supplementary and the 8 profiled runs were rerun at the clean development commit be19689
  (`docs/rerun_dependency_analysis.md`): no difference in any plan, cost or recorded
  Metropolis incumbent-cost trace (`results/processed/rerun_comparison.md`); Metropolis counts
  now include initialization; graph Metropolis rejections are now mostly `not_established`
  (F1); profiled VGG16 counts 1, 2, 4 are `unavailable`. Superseded raw outputs:
  `results/raw/archive_0892134/`.
* **Audit of commit ac66b9e.** A1: configuration ranking and DP scaling use the nominal
  planner's times and FLOPs (no raw-input gradient for the layers that consume the input
  image); equal-FLOP DP/MP median ratio 3.049 → 3.055; held-out validation unchanged. A2:
  ingestion guarantee narrowed to fingerprint-marker presence. A3: notebook v3 writes
  metadata only after the resume guard (CPU smoke test covers compatible and rejected
  resumes). A4: rerun checker compares seeds and recorded incumbent-cost traces, requires all
  25 pairs and exits nonzero on missing files: 25/25 pairs, 51 traces, 0 differences.
  Planner code and all experiment outputs unchanged.
* **Audit of commit e8f8d15.** Notebook v3: completed reduction measurements kept on a
  compatible resume; the resume fingerprint covers the recorded hardware class, driver, CUDA,
  framework and precision policy (5 new regressions in `tests/test_final_resume_review.py`).
  Planner code and all results unchanged.

Notebook v3 has been generated and smoke-tested on CPU only; it has not been run on a GPU.
The T4 data are from notebook v2.

## 1. Discrepancy check

Every defect reported in `LHAS_Decision_Answers_and_Code_Review.md` was reproduced on
commit 757bd4c before changing code (`scripts/diagnostics/review_repro.py`); none was
found to be incorrect:

| Review finding | Reproduction at 757bd4c |
|---|---|
| Greedy fallback used the final layer's charges | AlexNet `classifier_4`, MP16 (not allowed at the 1000-output layer): All-gather shard 256 000 bytes used instead of 1 048 576 |
| Assembled All-gather bytes aliased solely because a node hosts an MP consumer | ResNet-50 final interface: delivered 2048×7×7, FC input 2048; mask 0 at node 1 while 308 281 344 bytes are received vs an 8 388 608-byte FC reservation |
| BatchNorm buffers | counter treated as float32 and divided by p; saved statistics not counted |
| Unresolved evaluations treated as infinite | VGG16 N = 256 records: 100, 79 and 89 incomplete Metropolis evaluations; `pure_dp_best` ignored `not_completed`; chain OWT stepped down after failures |
| GPipe contract | all nine winners with rematerialization off; M ∈ {4, 8, 16, 32} only; closed-form fill/drain timing |
| Verification coverage | thresholds 1.5 M transmissions / 400 k routes without per-operation records |

Additional finding (same class as the ResNet defect): GoogLeNet branch-4 consumers
apply max pooling to the concatenated interface tensor, so an MP consumer's input
reservation cannot hold the delivered tensor either; corrected with rule C1.

Material scientific ambiguities: none required stopping a task. Two points are
recorded as decisions of this implementation rather than of the review:
(i) the typed-interface Metropolis evaluator marks a proposal `not_established` when
its fastest schedule combination fails the budget without a proof (D10), and such
proposals are rejected and counted like label-capped chain proposals (D14);
(ii) the GPipe contract's largest microbatch count (M = 32) is selected in most
GPipe results, so a larger M range could lower GPipe times; M was not extended
beyond the contract.

## 2. Changes

Details: `docs/change_log.md` (defects, baselines, assumptions, runners),
`docs/decision_table.md` (D1–D17 and where each is implemented),
`docs/memory_ledger.md` (corrected rules), `docs/verification.md` (coverage records).
Summary:

* **Memory ledger (D8):** aliasing only into a named reservation with matching
  elements (consumer input, J1 operand buffer, new C1 buffer), producer alias only for
  in-place ReLU maps; J1 3V for shape-changing consumer operators; identity holder
  (K+1)V; BatchNorm running and saved statistics and an int64 counter per owner.
* **Statuses:** `feasible` / `infeasible` (proven) / `not_established` / `incumbent`
  in both runners and in tables and figures; branch fixed plans use a reservation
  lower bound over all retained alternatives to prove infeasibility; best-common DP
  certified when unresolved counts are dominated by their lower bounds.
* **Greedy ablation:** current-layer local charge (`ChainModel.local_charge`).
* **Baselines:** one OWT definition, no repair; best-common DP with per-count
  statuses; multi-start FlexFlow-derived search with unresolved-proposal counting;
  GPipe per D13 (S = 1..ℓ, R ∣ B, P ≤ N, M ∈ {1..32}, rematerialization on/off,
  dependency simulation, circuit check, exact joint All-reduce selection).
* **Assumptions:** B = 1024, manifests and hashes, no auxiliary heads, SyncBN,
  raw-input gradient omitted, momentum state, 1 GiB reserve (`--reserve` in every
  runner, serialized).
* **Verification:** one record per selected operation with expected counts.
* **Runners and outputs:** schema `lhas-run-v2` with provenance and runtime split;
  graph and GPipe runners rewritten; queues (`scripts/final_queues.py`).
* **Colab (D17):** embedded manifests with a pre-profiling shape check, try/finally
  cleanup, incremental CSV with resume, minimum sample counts, signed and absolute
  errors with tails and Spearman correlation, reduction benchmark as a
  model-equivalent rate; ingestion rejects batch, manifest, precision and local-shape
  mismatches; the planner lookup checks the same.
* **Robustness fixes made during the runs:** an empty candidate set now raises
  (never a zero-cost operation); the Metropolis retry cap is bounded by label-matrix
  memory at N ≥ 512 (unchanged for N ≤ 256) after VGG16 N = 512 was stopped by the
  8 GB memory limit; the GPipe All-reduce selection uses a per-stage feasibility bound
  (the node limit is no longer reached; results unchanged); the Colab ingestion keeps
  the last row per shape, so a shape measured on a resumed notebook run is not also
  listed as unmeasured (commit 2bb17fe; the notebook itself is unchanged).

## 3. Tests executed

| Command | Result |
|---|---|
| `python -m pytest -q tests` (final, after the last code change) | **362 passed, 1 skipped** (363 collected; the skip is a (p, q) pair with p > N), 219 s |
| of which new: `tests/test_review_regressions.py` | 20 tests: greedy local charge; ResNet-50 final-interface pre-pooling operands; GoogLeNet C1; storage-preserving producer maps; BatchNorm state and int64 counter; SyncBN forward/backward equal to full-batch BatchNorm (PyTorch, float64, p = 1, 2, 4; MP channel-local); reserve once per node; label cap → `not_established`; incumbent and lower-bound certification; graph fixed-plan statuses; one OWT definition (N = 16, 64); multi-start Metropolis with unresolved counting and unfound starts; GPipe simulation = (M+S−1)(t_f+t_b), M ∈ {1, 2}, S = 1 flag, budget, cost = sum of parts; All-reduce ring translation; verification coverage records (actual, reduced, skipped) |
| `tests/test_profiled_mode.py`, `tests/test_colab_safeguards.py` | 8 tests: profiled lookup; rejection of B, manifest, precision, TF32 and local-shape mismatches; notebook logic and ingestion checks on CPU |
| `python colab/cpu_smoke_test.py` | passed: 136 exported layers match the manifests; 1783 unique local shapes; incremental CSV and resume; reduction benchmark; validation statistics (perfect-model check: 0 error, Spearman 1); kernel decomposition = autograd; ingestion accepts the untampered ZIP and rejects five tampered ones. No GPU timing. |
| `scripts/diagnostics/review_repro.py` at 757bd4c | reproduced every review finding (Section 1) |
| `python -m pytest -q` after the review-0892134 fixes | **374 passed, 1 skipped** (same skip), including `tests/test_review_0892134.py` (F1 fixture and GoogLeNet case, F3 ×2, T5, T6 chain and graph, F4 ×2, F5, T1) and the extended `tests/test_colab_safeguards.py` (notebook v3 resume guard; NaN timing, missing column, duplicate signature, missing fingerprint rejected; stored T4 table reproduced exactly) |
| `python -m pytest -q -rs tests` after the review-ac66b9e corrections (Python 3.11.15, numpy 2.4.4, numba 0.67.0, torch 2.14.0+cu130, torchvision 0.29.0+cu130, pytest 9.1.1) | **383 passed, 1 skipped** (participants > N), 153 s; log `results/logs/pytest_full_review_ac66b9e.log`. Focused (`test_review_ac66b9e.py`, `test_review_0892134.py`, `test_colab_safeguards.py`): 22 passed. `python colab/cpu_smoke_test.py`: passed, including compatible and rejected resume sessions (`results/logs/cpu_smoke_review_ac66b9e.log`). |
| `python -m pytest -q -rs tests` after the e8f8d15 audit (development commit 80f9b07; same environment) | **388 passed, 1 skipped** (participants > N); log `results/logs/pytest_full_final.log`. `python colab/cpu_smoke_test.py`: passed (`results/logs/cpu_smoke_final.log`). |
| Public checkpoint (this tree, 2026-10-04; see `PROVENANCE.md`) | `python -m pytest -q -rs tests`: **388 passed, 1 skipped** (participants > N), 250 s; `python colab/cpu_smoke_test.py`: passed; cheap reproduction (README) of AlexNet N = 64 with DP(N), best-common DP and OWT: identical costs, configurations, statuses and source digest; `analyse_profile.py`, `make_figures.py`, `make_tables.py`, `compare_provisional.py` rerun in a copy: every processed file identical, figure PDFs identical apart from their creation dates; `compare_reruns.py`: 25/25 pairs, 51 traces, 0 differences. |
| Verification of every selected LHAS operation (inside the runs) | all 17 main plans (N = 64–512, AlexNet 1024): every network operation `checked-actual` with packing re-derived; no reduced-payload or skipped operation (table in `results/processed/tables.md`) |

## 4. Experiments completed

All runs use `LHAS_PACK_CACHE_DIR=results/cache/packings` (packings keyed by transport
parameters and route sets, unchanged by the final specification), except the
cold-cache runs. Output schema `lhas-run-v2`; logs in `results/logs/`.

| Experiment | Runs (all exit 0 in their final form) | Output |
|---|---|---|
| Main comparison, N = 64–512 | 16 (4 workloads × 4 N), all baselines, 3 × 1000 Metropolis iterations, verification | `results/raw/final/` |
| AlexNet N = 1024 (supplementary) | 1 + GPipe 1 | `final_supp/`, `final_supp_gpipe/` |
| GPipe (AlexNet, VGG16) | 8 | `final_gpipe/` |
| Planner ablations (N = 64, 256) | 4 | `final_ablations/` |
| Collective families (N = 64) | 12 | `final_family/{all,one-stage,deepest}/` |
| Batch sensitivity B = 256 (N = 64) | 4 | `final_batch/B256/` |
| One-at-a-time sensitivity (N = 64) | 92 (23 settings × 4 workloads) | `final_sens/` |
| Cold packing cache (runtime) | 5 | `final_cold/` |

Stops and reruns (recorded in the logs): VGG16 N = 512 was stopped by the 8 GB
memory limit (exit 137) in the Metropolis retry; it was rerun after bounding the
retry cap by label-matrix memory at N ≥ 512 (no N ≤ 256 run is affected; AlexNet
N = 512 never reached the retry). Three runs were stopped deliberately (exit 143) to
apply the neutral unresolved-proposal status and the lower-bound certification of
best-common DP before they wrote results, or to reschedule for memory; AlexNet
N = 64/128, GoogLeNet N = 64–512 and VGG16 N = 64/128 were rerun so that every main
result uses the final baseline logic; the GPipe runs were rerun after the exact
All-reduce selection was tightened (values unchanged; the two earlier `incumbent`
statuses became `feasible`). Each output records its code commit and a dirty flag;
three runs (AlexNet N = 512, ResNet-50 N = 64, VGG16 N = 256) started with uncommitted source changes that were committed immediately
afterwards (see `provenance.code_dirty`). After the review of 0892134, all primary,
supplementary and profiled runs were rerun at the clean commit be19689 (queues
`results/queues/rerun_{A,B}.txt`, all exit 0); outputs now also record `source_sha256`.

Added after the Colab run: profiled planning sensitivity with measured Tesla T4
computation, N = 64 and 256, 4 workloads, LHAS, DP(N), best-common DP, OWT (8 runs,
`results/raw/final_profiled_T4/`, all exit 0; Section 8).

Not run: the optional N = 1024 runs of VGG16, GoogLeNet and ResNet-50.

## 5. Results (final specification)

Predicted time per iteration (ms). † incumbent; ‡ Metropolis search with rejected
unresolved proposals (VGG16: 175–251 of 2 671–2 864 evaluations per system size, summed
over seeds with initialization included, label cap; GoogLeNet/ResNet-50: at most 526,
fastest-combination failures without proof, counted as not established since F1). All LHAS plans are `feasible`; for GoogLeNet and ResNet-50
the D10 certificate holds (exact additive minimizer within the budget) at every N.

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

Main observations (details in `results/processed/tables.md`):

* AlexNet and VGG16: CONV layers DP, FC1–2 MP on 8 nodes, FC3 on one node (VGG16 at
  N = 512 also lowers the last three CONV counts to 128/128/64). LHAS vs best-common
  DP: 2.73–3.05× (AlexNet), 1.92–2.58× (VGG16); vs OWT 1.55–7.87× and 1.14–3.98×;
  vs GPipe 1.78–1.81× and 1.39–2.00×.
* GoogLeNet: DP64 for all CONV layers, final FC on one node; 1.04× vs best-common DP.
  ResNet-50: pure DP at the best common count (1.00×). Gains over DP(N) at large N
  come from using fewer nodes.
* FlexFlow-derived search: equal to LHAS in 6 of 16 cases, within 1.12× otherwise.
* Planner ablations: strategy-only (= OWT) 1.14–2.98× slower; DP-only 1.20–1.63×;
  MP-only 6.8–7.5× (AlexNet) or infeasible (VGG16); corrected greedy equal to LHAS
  except AlexNet N = 256 (1.14×).
* Collective families at N = 64: one-stage 1.01–1.14×, deepest 1.49–2.16× slower.
* Sensitivity at N = 64: α = 1 −18 to −31%; ρ = 0.25 +35 to +74%; ε = 0.5 +22 to
  +54%; t_reconf = 100 µs up to +35% (GoogLeNet); λ = W from 32 to 256 reduces the time by 2.5–8.9%;
  no reach/peer limit −4 to −29%; whole items up to +24%; raw-input gradient kept
  ≤ +1%; h = 0 and no momentum: plans unchanged. LHAS never slower than DP(N) or OWT.
* B = 256 at N = 64: LHAS vs best-common DP 4.22× (AlexNet), 3.86× (VGG16),
  1.06× (GoogLeNet), 1.03× (ResNet-50).
* Runtime (reruns): AlexNet N = 512 construction 282 s / search 68 s (reused cache) vs
  788 s / 115 s (empty cache); GoogLeNet N = 512 typed-interface search 1702 s;
  Metropolis up to 34 min per seed (VGG16). Wall clock on a shared 2-core container.

## 6. Numerical changes relative to the provisional results (757bd4c)

Provisional results are archived unchanged in `results/archive/provisional_757bd4c/`
(never relabeled). Cell: provisional → final (ratio). Full table:
`results/processed/provisional_vs_final.md`.

| workload | N | LHAS | DP (p=N) | DP best | OWT | FlexFlow-MCMC | GPipe | LHAS configs changed |
|---|---|---|---|---|---|---|---|---|
| alexnet | 64 | 65.7 → 65.2 (0.99) | 239.0 → 238.6 (1.00) | 202.4 → 198.5 (0.98) | 101.3 → 100.8 (1.00) | 71.4 → 65.2 (0.91) | 123.2 → 116.2 (0.94) | no |
| alexnet | 128 | 66.8 → 66.5 (1.00) | 276.0 → 275.7 (1.00) | 202.4 → 198.5 (0.98) | 133.2 → 132.9 (1.00) | 71.5 → 73.5 (1.03) | 206.4 → 120.0 (0.58) | no |
| alexnet | 256 | 73.1 → 72.6 (0.99) | 298.0 → 297.9 (1.00) | 202.4 → 198.5 (0.98) | 216.8 → 216.6 (1.00) | 78.7 → 74.6 (0.95) | 248.5 → 131.2 (0.53) | no |
| alexnet | 512 | 73.1 → 72.6 (0.99) | 500.0 → 499.9 (1.00) | 202.4 → 198.5 (0.98) | 571.6 → 571.6 (1.00) | 87.7 → 77.9 (0.89) | 283.6 → 131.2 (0.46) | no |
| alexnet | 1024 | 73.1 → 72.6 (0.99) | 587.0 → 587.0 (1.00) | 202.4 → 198.5 (0.98) | 1635.6 → 1635.6 (1.00) | 93.3 → 77.1 (0.83) | 332.6 → 131.2 (0.39) | no |
| vgg16 | 64 | 456.5 → 455.9 (1.00) | 874.0 → 875.1 (1.00) | 874.0 → 875.1 (1.00) | 518.7 → 519.8 (1.00) | 456.9 → 455.9 (1.00) | 774.9 → 634.7 (0.82) | no |
| vgg16 | 128 | 323.5 → 323.2 (1.00) | 760.5 → 760.2 (1.00) | 760.5 → 760.2 (1.00) | 453.0 → 452.7 (1.00) | 336.4 → 353.0 (1.05) | 824.6 → 492.8 (0.60) | no |
| vgg16 | 256 | 283.0 → 282.8 (1.00) | 730.7 → 730.6 (1.00) | 730.7 → 730.6 (1.00) | 580.7 → 580.6 (1.00) | 303.5 → 306.1 (1.01) | 684.1 → 566.3 (0.83) | no |
| vgg16 | 512 | 353.0 → 352.8 (1.00) | 1129.7 → 1129.6 (1.00) | 787.4 → 787.1 (1.00) | 1402.8 → 1402.7 (1.00) | 378.8 → 394.9 (1.04) | 663.8 → 659.6 (0.99) | no |
| googlenet | 64 | 84.0 → 83.2 (0.99) | 87.5 → 86.7 (0.99) | 87.5 → 86.7 (0.99) | 90.3 → 89.5 (0.99) | 84.3 → 83.5 (0.99) | — → — | no |
| googlenet | 128 | 95.3 → 94.6 (0.99) | 100.0 → 99.6 (1.00) | 99.0 → 98.2 (0.99) | 110.2 → 109.8 (1.00) | 95.5 → 94.8 (0.99) | — → — | yes |
| googlenet | 256 | 95.4 → 94.6 (0.99) | 143.0 → 142.8 (1.00) | 99.0 → 98.2 (0.99) | 178.9 → 178.7 (1.00) | 137.1 → 96.1 (0.70) | — → — | no |
| googlenet | 512 | 95.4 → 94.6 (0.99) | 301.9 → 301.8 (1.00) | 99.0 → 98.2 (0.99) | 428.8 → 428.7 (1.00) | 287.0 → 96.1 (0.34) | — → — | no |
| resnet50 | 64 | 236.5 → 235.7 (1.00) | 236.5 → 235.7 (1.00) | 236.5 → 235.7 (1.00) | 293.7 → 292.8 (1.00) | 236.5 → 235.7 (1.00) | — → — | no |
| resnet50 | 128 | 246.5 → 246.1 (1.00) | 246.5 → 246.1 (1.00) | 246.5 → 246.1 (1.00) | 450.1 → 449.7 (1.00) | 246.5 → 246.1 (1.00) | — → — | no |
| resnet50 | 256 | 253.2 → 252.4 (1.00) | 321.6 → 321.4 (1.00) | 253.2 → 252.4 (1.00) | 1503.2 → 1503.0 (1.00) | 321.6 → 252.4 (0.78) | — → — | no |
| resnet50 | 512 | 253.2 → 252.4 (1.00) | 643.4 → 643.3 (1.00) | 253.2 → 252.4 (1.00) | 7234.0 → 7233.9 (1.00) | 643.4 → 252.4 (0.39) | — → — | no |

* LHAS, DP and OWT times change by at most about 1–2% (mainly the omitted raw-input
  gradient). The memory corrections, momentum state and reserve change no selected
  LHAS plan except GoogLeNet N = 128 (all CONV layers DP128 → DP64, 95.3 → 94.6 ms).
* FlexFlow-derived search: large improvements at large N from the multi-start
  initialization (ResNet-50 N = 512: 643.4 → 252.4 ms; GoogLeNet N = 512:
  287.0 → 96.1 ms); small changes elsewhere, in both directions (randomized search,
  new evaluator rules).
* GPipe: 0.39–0.99× of the provisional values (contract D13: M from 1 to 32,
  active-node and replica search, exact All-reduce selection, dependency simulation).
* Planner ablation: the provisional 13% greedy gap came from the defective greedy;
  the corrected greedy matches LHAS in 3 of 4 cases (1.14× at AlexNet N = 256).
* Results reported for earlier versions of this work (speedups, collective sensitivity and
  FC node counts) are not reproduced by the regenerated model and were not used as targets.

## 7. Remaining limitations

* **Model-based.** No optical-hardware or independent-simulator validation. The
  computation model is analytical (hypothetical P100, ρ = 0.5). On a Tesla T4, a
  constant-throughput model of that form mispredicts held-out kernel signatures (median
  absolute relative error 59%, 57–64% over splits), cannot order DP(p) and MP(p) of equal
  FLOPs, and overstates the efficiency of local batches ≤ 4. With measured T4 computation
  at modeled N = 64 and 256, the LHAS–DP–OWT ordering persists; this does not isolate
  constant utilization, validate distributed execution or retest Metropolis and GPipe.
* **Objective scope.** The branch objective is additive, not a concurrent makespan;
  exactness is within the candidate families. No overlap in the nominal model.
* **Memory.** Conservative accumulated reservations, not lifetimes or measured peaks;
  the 1 GiB reserve is an assumption.
* **Baselines.** Adaptations to the same model. The Metropolis search rejects
  unresolved proposals (counts disclosed). GPipe uses a restricted partition, and
  M = 32 (the contract's maximum) is selected in 8 of 9 cases, so a larger M range
  could lower GPipe times. AccPar, Megatron-style and sharded systems are not compared.
* **Coverage.** Sensitivities, families and B = 256 at N = 64 only; ablations at
  N = 64 and 256. Other N = 1024 runs not done.
* **Runtime figures** are wall clock on a shared 2-core container with 2–3
  concurrent runs; indicative only.
* **Profiled planning.** 24 T4 shapes (early VGG16 CONV layers at large local batch
  or full-batch MP; 31 layer/configuration entries) were not measured and are excluded
  for every method in the profiled runs; profiled planning was run at N = 64 and 256
  only, combines T4 computation with the modeled optical network, and is not a
  measurement of a T4 cluster.
* `configs/approved_reference_parameters.json` (frozen, author-approved) still carries
  its original status note about unpinned inputs; `configs/nominal_experiment.json`
  now pins them.
* **Release.** Released under the MIT License (`LICENSE`); no archival release (for example
  Zenodo) has been made. Provenance of the public tree: `PROVENANCE.md`; outstanding work:
  `docs/STATUS.md`.

## 8. Measured-GPU validation (Tesla T4, run by the author with notebook version 2)

Files: `results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip` (unmodified; notebook v2),
ingested to `configs/profiles/profiled_Tesla_T4_B1024.json`. Protocol:
`docs/t4_validation_protocol.md` (grouped by timed kernel signature after review 0892134, T1;
ranking and scaling with the planner's times after review ac66b9e, A1). The text below is
`results/processed/t4_validation.md`, generated with the figure from
`results/processed/t4_validation.json` by `python scripts/analyse_profile.py`.

Source: `results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip` (sha256 e1648ca8e6dbb6a1…; notebook v2); GPU Tesla T4, CC 7.5, driver 580.82.07, CUDA 12.8, cuDNN 91900, torch 2.11.0+cu128, torchvision 0.26.0+cu128; float32, TF32 off; manifest check: match. Local computation only; no optical, multi-GPU or iteration-time measurement. Protocol: `docs/t4_validation_protocol.md`.

### Coverage

* Profiled records (shape IDs): 1783; measured 1759; not measured (memory cap): 24; repeated IDs removed: 0.
* Distinct timed kernel signatures: 1542 measured (1566 over all records). 114 signatures were timed more than once (331 records), because the notebook keyed records by the parent width too.
* Layer/configuration entries: 2770 measured of 2801. Unmeasured: vgg16/features_0 (4), vgg16/features_12 (2), vgg16/features_14 (2), vgg16/features_2 (10), vgg16/features_5 (3), vgg16/features_7 (10).
* Raw-sample consistency: 21108 summary values recomputed from the raw samples, 0 mismatches.
* Repeatability, repetitions within one session on one GPU (30 after 10 warm-up iterations); not run-to-run or device-to-device variation. Median IQR/median forward 3.1% (p90 20.4%), input_grad 3.0% (p90 14.4%), weight_grad 3.7% (p90 19.6%).

### Constant-throughput model, grouped split (no shared signatures)

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

### Configuration ranking within a layer (split-independent)

Over 136 layers with at least two measured configurations, using the time the planner reads for each configuration:

* Pairs with different local FLOPs: 26393; measured order agrees with the FLOP order in 90.5% (alexnet 89.1%, googlenet 90.5%, resnet50 91.1%, vgg16 89.6%).
* Same strategy, consecutive measured counts: 2500 pairs; 91.1% measured faster at the larger count.
* DP(p) and MP(p) pairs with equal local FLOPs (the constant-throughput model cannot order them): 1066; measured slower/faster ratio median 3.05, 90th percentile 17.99; ratio above 1.25 in 75.8%; DP faster in 83.8% (893), MP faster in 43; 130 pairs are p = 1, where DP and MP time the same kernel (ratio 1). For the 936 pairs with p > 1: median ratio 3.77, 90th percentile 18.59.

Global Spearman correlation measures ordering across shapes of very different size; the within-layer figures above measure the ordering among competing configurations of one layer.

### Direct-FLOP-equivalent rate by shape size (one point per measured signature)

| 3 × FLOPs per product | signatures | median (TFLOP/s) | 10th–90th percentile | share of measured time |
|---|---|---|---|---|
| 1e+06–1e+08 | 98 | 0.18 | 0.06–0.35 | 0.000 |
| 1e+08–1e+09 | 247 | 0.80 | 0.20–1.63 | 0.004 |
| 1e+09–1e+10 | 484 | 1.92 | 0.19–4.03 | 0.056 |
| 1e+10–1e+11 | 457 | 1.86 | 0.55–5.34 | 0.227 |
| 1e+11–1e+12 | 219 | 2.99 | 1.38–6.34 | 0.396 |
| 1e+12–1e+14 | 37 | 5.14 | 3.30–8.61 | 0.317 |

70 signature products exceed the FP32 peak when their FLOPs are counted as direct convolutions (conv 3x3 stride 1: 43, conv 5x5 stride 1: 27). This is consistent with cuDNN selecting Winograd or FFT algorithms; the algorithms were not identified by measurement.

### DP scaling efficiency by local batch

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

### Local reduction

median model-equivalent rate (a+2)V/t of the fused sum, V >= 16 MiB: 218.6 GB/s (12 measurements). The rate at which the planner's reduction term reproduces the measured time; not a DRAM bandwidth measurement.

### Superseded: the notebook's record-ID split (documents finding T1)

879 calibration and 880 validation records; 71 signatures on both sides, 116 validation records with a calibration twin. Recomputed from the CSV (agrees with the notebook's JSON: True): median 69.2%, p90 99.0%, p95 143.2%, maximum 414.0%, signed median -39.1%, weighted 81.0%, summed 30.2%, Spearman 0.886. Not used for any claim.

Cross-check of the reviewer's illustrative string-typed grouped split: median 60.6% on 771 validation signatures.


### Profiled planning sensitivity (measured T4 computation, T4 reduction rate 218.6 GB/s; reruns)

LHAS time (ms) and baseline/LHAS ratios; analytical P100-reference values in parentheses
(full candidate set).

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

Plans keep their structure (CONV DP; FC1–2 MP on 8 nodes for AlexNet and VGG16; final
FC on 8 nodes instead of 1; AlexNet CONV on 32 instead of 64 nodes at N = 256;
GoogLeNet and ResNet-50 essentially DP). Every selected network operation of these plans
was verified at its actual payload. This is a sensitivity study at modeled ring sizes; it
changes computation inputs, reduction rate and candidate set together and says nothing
about distributed execution.
