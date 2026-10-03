# Verification: what is checked, and what kind of evidence it is

The categories are kept separate, as requested.

| Category | What it establishes | What it does not establish |
|---|---|---|
| **Mathematical checks** | Element-level ownership of every boundary matrix: each destination gets exactly its required elements, nothing is received twice, and sources send only what they hold. The closed forms S/(p_i p_j) and S/p_i hold when divisibility holds. The relay reservation bound covers actual chunk occupancy. The exhaustive enumeration optimum equals the DP optimum. | Anything about hardware |
| **Implementation-consistency checks** | The schedule constructor, the numba timing and buffer kernels, and the planners agree with the written model. They also agree with an independent re-implementation (`lhas_verify`, no imports from `lhas`) and with a second, independently assembled cost engine (typed interfaces vs chain). The four-node worked example reproduces the approved table exactly. | That the model is physically accurate |
| **Independent simulation** | None. No external simulator (e.g., ASTRA-sim) or event-driven network simulation was used. | — |
| **Hardware measurements** | None of the optical model. Local computation on one Tesla T4 (Colab, run by the author on 2026-09-28): 1759 of 1783 profiled records measured, 1542 distinct timed kernel signatures. On validation signatures of a grouped split (no signature shared with calibration; `docs/t4_validation_protocol.md`), the constant-throughput model has a median absolute relative error of 59% (57–64% over 100 splits), a time-weighted absolute error of 63% and a global Spearman correlation of 0.90; it cannot order DP(p) and MP(p) partitions of equal FLOPs, whose measured times differ by a median factor of 3.1 over 1,066 pairs (130 of them p = 1, identical kernels; DP faster in 893 of the other 936) (`results/processed/t4_validation.md`). | Optical timing; P100 timing; distributed execution; iteration time of any network |

## Independent verifier (`src/lhas_verify`)

Each rule is checked on exported schedule records:

* **R1**: label range, reach, hop chaining.
* **R2**: routing rule (arc, direction, fixed-stride relays).
* **R3**: directed-segment/label conflicts, per-label Tx/Rx shared across directions, aggregate budgets, peer cap.
* **R4**: deterministic first-fit and E-A dissemination rules, against a pure-Python reference.
* **R5**: chunk partition with actual tail, step rule a+j−1, relay causality, one chunk per circuit per step, whole items in direct-only rounds.
* **R6**: availability at round start and final ownership/delivery.
* **R7**: All-reduce contribution multiplicity and declared reduction counts.
* **R8**: setup flags from the retained configuration.
* **R9**: step durations, transport, local reduction.
* **R10**: relay occupancy ≤ reservation, and no understated reservation.

Tests: `tests/test_verifier_valid.py` covers 102 constructor-output cases across five stressed parameter sets. These include:

* all unequal-group families;
* nonconsecutive representatives [1,4,7,10];
* multi-round packing, relays, multi-chunk items and short tails;
* whole-item forwarding, unrestricted reach and no peer cap;
* random All-to-all matrices.

`tests/test_verifier_rejects.py` feeds deliberately invalid schedules, each of which must be rejected with its rule code:

* label 65 with λ=64;
* reach exceeded;
* broken hop chain;
* longer-arc route;
* segment/label conflict;
* per-label Tx reused in opposite directions;
* budget exceeded;
* peer cap exceeded;
* merged rounds;
* first-fit nonconformance (higher label; needless deferral);
* dissemination target order and sender;
* chunk forwarded before reception;
* causal but off-rule step;
* rounded-up tail;
* split item in a direct-only round;
* two chunks on one circuit in one step;
* shard not held;
* shard forwarded in the round it is received;
* missing delivery;
* duplicate delivery;
* wrong All-to-all blocks;
* self-transfer;
* double-counted contribution;
* wrong reduction count;
* missing distribution;
* setup wrongly skipped;
* reuse not credited;
* superset or relabeled reuse;
* wrong timing claims;
* understated relay, gathered, retained and accumulator reservations.

## Verification of selected plan operations (coverage records)

Every run of `experiments/run_chain.py` and `experiments/run_graph.py` re-verifies
the selected LHAS plan with the independent verifier (`lhas/verify_plan.py`;
disable only with `--no-verify`). Each selected operation receives exactly one
record, and the output reports the expected operation count:

| Record status | Meaning |
|---|---|
| `checked-actual` | the schedule is rebuilt from its candidate identifier and verified at its actual payload |
| `checked-reduced` | verified at a reduced payload of 2κ+1 bytes per item (three chunks with a short tail) because the explicit hop-chunk transmission count at the actual payload exceeds `MAX_TRANSMISSIONS` = 1 500 000; the reason is recorded |
| `skipped` | not verified because the operation has more than `MAX_VERIFY_ROUTES` = 150 000 routes (resource limit of the pure-Python verifier); the reason is recorded |
| `local` | one participant; no network schedule |

`packing_checked` states whether the deterministic first-fit was re-derived by the
pure-Python reference (operations with ≤ `MAX_PACKING_ROUTES` = 20 000 routes).
`summary.all_network_operations_checked` is true only if no network operation was
skipped; it does not imply actual-payload or packing checks for every operation.
The coverage of every retained run is tabulated in `results/processed/tables.md`.
A failed check raises and the run is marked failed; it is never recorded as skipped.

**Two defects found by these checks and fixed:**

1. **κ = 0 timing.** The kernel reused the last route's full-chunk value per key, so whole-item (κ = 0) rounds were mis-timed when item sizes differed. Found by R9 on random All-to-all matrices.
2. **Verifier arc check.** The verifier's routing check compared node lists only. A single-hop route over the longer arc therefore passed. It is now compared hop by hop, including direction and segment count.

## Planner checks

* `tests/test_planner_exhaustive.py`: the DP matches brute force over every configuration sequence and **every** schedule combination, including terminal alternatives. Budgets are swept from infeasible to slack, with α ∈ {0, 0.7}. Cases where memory binds are asserted to occur, and a case where the fastest plan is infeasible but a slower schedule fits is constructed explicitly.
* `tests/test_graph_engine.py`: the typed-interface engine equals the chain DP on AlexNet and VGG16, for both raw-input-gradient settings. Its search objective also equals an independent re-evaluation of the returned plan.

## Pruning rules used by the planners (all exact)

* Pareto dominance over (cost, reservation vector), as in the manuscript.
* Safe coordinates: a coordinate v is replaced by 0 when μ_v plus the largest possible future reservation fits. The cheapest fully safe label then dominates all labels of no smaller cost.
* Infeasibility lower bound: μ plus the smallest possible future reservation must fit.
* Branch and bound: labels whose cost plus the admissible unconstrained cost-to-go exceeds an incumbent are dropped.
* Exactness shortcut: if the unconstrained minimizer is memory-feasible, it is optimal.

None of these is a label cap. Baseline fixed-plan evaluations and Metropolis proposals of the chain runner use explicit label caps (fixed plans: 50 000 labels; Metropolis proposals: 50 000, retried once at min(200 000, 51.2·10⁶/N) labels). A capped evaluation is reported as `not_established`, never as infeasible, and it is never cached as infinite cost: the best-common DP result becomes an `incumbent` and the Metropolis search is labeled resource-limited with the count of such evaluations. The typed-interface (branch) runner uses no label cap.

## Regression tests for the review defects (`tests/test_review_regressions.py`)

Greedy ablation charge of the current layer; ResNet-50 final-interface pre-pooling
operands; GoogLeNet consumer-side pooling (C1); storage-preserving producer maps
only; BatchNorm state bytes and the int64 counter; synchronized BatchNorm forward
and backward equal full-batch BatchNorm (two 2C reductions under DP, channel-local
under MP; compared with PyTorch in float64); runtime reserve charged once per node;
label-cap stop is `not_established`; incumbent semantics; graph fixed-plan
statuses; one OWT definition in both runners; multi-start Metropolis search with
unresolved-proposal counting; GPipe schedule simulation equals (M+S−1)(t_f+t_b)
for equal stages without transfers, microbatch counts 1 and 2, S = 1 degenerate
flag, budget respected and cost = sum of its parts; All-reduce ring translation
invariance used by GPipe; verification coverage records including reduced and
skipped operations. `tests/test_profiled_mode.py` and
`tests/test_colab_safeguards.py` cover the profiling safeguards (D17).
