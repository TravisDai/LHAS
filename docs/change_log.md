# Change log: model equations → implementation

The repository is a new implementation of the approved analytical model of the LHAS
paper (handoff of 2026-09-28). The manuscript is not part of this repository until
publication. The repository contains no code from the lost original simulator. Each row
gives the module that implements a numbered equation or rule of the model (identified by
its label in the manuscript source), and the tests that exercise it.

| Model item (manuscript label) | Implementation | Tests |
|---|---|---|
| Eq. `bg_feasible_nodes` (p ∣ B for DP, p ∣ n_i for MP) | `model/chain.py: ChainModel.configs`, `model/branch.py: GraphModel.configs` | `test_planner_exhaustive.py`, `test_graph_engine.py` |
| sec `optical_model`: labels, shared per-label Tx/Rx, W budgets, reach 8, peer cap 16, routing (shorter arc, CW ties, fixed-stride relays) | `transport/schedule.py: route_geometry, hop_nodes`; `transport/kernels.py: pack_first_fit` | verifier R1–R3 (`test_verifier_valid.py`, `test_verifier_rejects.py`) |
| sec `transport`: priority order, deterministic first-fit, route admission/undo, rounds | `schedule.py: priority_order, build_phase`; `kernels.py: pack_first_fit` (Fenwick saturation prefilter, exactness-preserving; `pack_first_fit_plain` reference) | verifier R4 (independent pure-Python first-fit); kernel equivalence on 300 random instances (`scripts`/session log) |
| Eq. `chunks`; step rule a+j−1; direct-only rounds send whole items | `kernels.py: round_body_times`; `transport/export.py` (explicit transmissions) | verifier R5 incl. tail, multi-chunk, causality, direct-only splitting |
| Eq. `bg_generic_collective` (step durations, setup r_τ) | `kernels.py: round_body_times`; `schedule.py: evaluate` | verifier R8/R9; worked example |
| Circuit reuse: retained configuration, subset rule, operation starts unconfigured | `schedule.py: round_circuit_keys, assign_setup_flags` | verifier R8 (`test_setup_reuse_subset_allowed_and_misclaims_rejected`) |
| Relay reservation 2 min(κ, V) at every physical relay | `kernels.py: relay_peaks` | verifier R10 (occupancy ≤ reservation; claims not understated) |
| Eq. `local_reduction` | `schedule.py: evaluate` | verifier R7/R9 |
| Eq. `one-stage` / adapted OSM (one phase, one item per nonzero entry) | `collectives/families.py: osm` | `test_alltoall_random`, rejection tests |
| Eqs. `bg_wrht_group_constraints`, `wrht6`: WRHT m ∈ {2..min(p,2L+1)} ∪ {p}, collect and exchange top modes | `families.py: wrht_levels, wrht_m_values, wrht_op, wrht_ops` | `test_wrht_candidates`, worked example |
| OpTree adapted family (canonical mixed-radix factorization, list positions, actual holdings) | `families.py: mixed_radix_factors, mixed_radix_stages, optree_ops` | `test_nonconsecutive_representatives`, worked example |
| Eqs. `bg_osm_wavelengths`, `bg_osm_steps`, `bg_optree_steps`, `bg_optree_stage_steps`, `bg_optree_endpoint_bound`, `optree6`, `bg_optree_choice` | reference bounds only; **not used as costs** (costs come from constructed schedules) | — |
| Eq. `unequal_groups`, C-A (both regimes), C-B, C-C, E-A (multi-source dissemination), E-B, E-C | `families.py: contraction_group_size, allgather_ops`; `schedule.py: build_dissemination`; `kernels.py: pack_dissemination_round` | `test_allgather_candidates` (12 (p,q) pairs × 5 parameter sets), verifier R4 dissemination rule |
| Eqs. `T1`, `T2`, `T3`; tie rule; nondominated retention | `collectives/service.py: CollectiveService, pareto, _tie_key` | `test_planner_exhaustive.py` |
| Eqs. `intra_mp_payloads`, `intra_dp_payload` (with U_i-based All-gather payload, bias) | `model/chain.py: _intra_ops`; `model/branch.py: _own_ops, charge` | `test_graph_engine.py` |
| Eqs. `boundary_mp_dp_bp_payload`, `boundary_alltoall_payload`, `boundary_dp_mp_bp_payload` (F_{i,b}), `dd_matrix`, `mm_matrix` | `model/layout.py` | `test_layout.py` (element-level ownership, 180 cases) |
| Eqs. `intra_cost`, `inter_cost`, `comm` | `model/chain.py: transition` | `test_graph_engine.py` (two independent assemblies agree) |
| Eq. `comp` (2 FLOPs/MAC, three products; raw-input-gradient flag) | `model/workload.py: WLayer.mult_flops_per_example`; `model/chain.py: flops, tcp, needs_input_grad` | `test_raw_input_gradient_flag_changes_first_layer_only` |
| Eq. `layer_charge` (overlap α on intra transport only) and Eq. `terminal_charge` | `model/chain.py: TransAlt.G`; `model/branch.py: charge` | exhaustive tests with α ∈ {0, 0.7} |
| Eqs. `memory_dp`, `memory_mp`, `memory_feasibility`; e_i, e_ℓ^term | `model/layercost.py: layer_reservation, norm_state_bytes`; `model/chain.py: _assembled_mask, _u_storage`; `model/branch.py: ag_mask, consumer_full_T_buffer`; `planner/graph_eval.py`; `docs/memory_ledger.md` | exhaustive tests with binding budgets; `test_review_regressions.py` |
| Eqs. `training_objective`, `optimal6`, `memory_update`, `memory_dp_recurrence`; Algorithm `alg:optical-config` | `planner/chain_dp.py: plan_chain` (labels, Pareto dominance, safe coordinates, branch-and-bound on admissible bounds) | `test_planner_exhaustive.py` (DP = brute force over all sequences and all schedule combinations) |
| sec `branch`: typed interfaces, Eq. `branch_additive` (additive proxy), composite regions | `model/graph.py: build_structure`; `model/branch.py`; `planner/graph_dp.py`; `planner/graph_eval.py` | `test_graph_engine.py` |
| sec `reference_compute` and parameters file | `params.py: load_approved` | all |

## Revision of 2026-09-28: final specification (D1–D17) and review corrections

Source: `LHAS_Decision_Answers_and_Code_Review.md`. Every review finding was
reproduced at commit 757bd4c before it was changed
(`scripts/diagnostics/review_repro.py`); none was found to be incorrect. One
additional instance of the aliasing defect was found (GoogLeNet consumer-side max
pooling, below).

### Implementation defects corrected

| Defect (review) | Correction | Regression test |
|---|---|---|
| Greedy ablation fell back to `terminal(c)`, i.e. the final layer's All-gather/All-reduce payloads, for a candidate not allowed at the next layer (AlexNet MP16 at `classifier_4`: 256 000-byte shard instead of 1 048 576) | `ChainModel.local_charge(i, c)`: the layer's own computation and intra-layer operations with same-prefix gathering, independent of the next layer | `test_greedy_local_charge_uses_its_own_layer_payloads` |
| A1 zeroed assembled All-gather bytes at every MP consumer node (ResNet-50 final interface: 308 281 344 received bytes vs an 8 388 608-byte FC input) | Aliasing only into a named reservation with matching elements: consumer input `x` (no consumer operators, equal numel), the J1 operand buffer, or the new C1 buffer; producer alias only for in-place ReLU maps | `test_resnet_final_interface_charges_prepooling_operands` |
| (additional) GoogLeNet branch-4 max pooling at the consumer: same unjustified alias | C1: 2V (T and its gradient) at consumers with consumer-side operators | `test_googlenet_consumer_side_pooling_is_charged_C1` |
| Dropout treated as an identity Φ (producer alias) | `STORAGE_PRESERVING_OPS = {relu}`; dropout, BN, pooling, flatten store separately | `test_producer_alias_only_for_inplace_relu` |
| J1: one extra operand buffer regardless of consumer operators | shape-preserving add: V (or 0 when the identity is read in place from an identical layout); shape-changing consumer operators: 3V (pre-pooling operands and gradient) | ResNet test above |
| B1 identity holder (K+2)·V double-counted its own contribution | (K+1)·V | — |
| BatchNorm buffers: `num_batches_tracked` treated as float32 and divided by p; saved statistics not counted | running mean/var 2C·ψ and saved statistics 2C·ψ (÷p under MP), int64 counter 8 bytes per owner node | `test_batchnorm_state_bytes_and_int64_counter` |
| Statuses conflated (label-cap stops treated as infeasible, `pure_dp_best` ignored them, OWT stepped down after them) | `feasible` / `infeasible` (proven) / `not_established`; `incumbent` for best-of-set with unresolved members; branch fixed plans use a reservation lower bound over all retained alternatives to prove infeasibility | `test_label_cap_is_not_established_not_infeasible`, `test_summarize_counts_incumbent_semantics`, `test_graph_fixed_plan_statuses` |
| Verification: operations skipped or payload-reduced without per-operation records | one record per selected operation (`checked-actual`, `checked-reduced`, `skipped`, `local`) with reasons, packing-check flag and expected-operation count; chain and branch plans | `test_verification_records_every_selected_operation` |
| Family restriction could in principle leave an empty candidate set, which would have been charged as a zero-cost operation | empty candidate sets raise | — |
| Collective prefetch evaluated eager candidate lists per payload | one lazy candidate stream per structure for all payloads (identical results, lower memory) | full suite |

### Baselines repaired

| Baseline | Change |
|---|---|
| OWT (D12) | one definition (`baselines/common.owt_configs`) in both runners; the chain step-down repair removed |
| Best-common DP (D11) | per-count statuses; incumbent when an unresolved count is not dominated by its lower bound (unconstrained optimum or min-charge objective); both runners |
| FlexFlow-derived search (D14) | starts: best-common DP, adapted OWT, DP(N), a feasible random plan; best start preserved; unresolved proposals rejected and counted (chain: label cap 50 000, retried once at min(200 000, 51.2·10⁶/N)), never cached as infinite; traces of incumbent vs evaluations and time |
| GPipe (D13) | S = 1..ℓ (S = 1 degenerate DP), R ∣ B with P = R·S ≤ N and explicit node mapping, M ∈ {1, 2, 4, 8, 16, 32}, rematerialization on and off, dependency/resource schedule simulation, simultaneous boundary-circuit check, momentum and reserve, exact joint selection of nondominated All-reduce alternatives |
| Graph baselines | `baselines/graph_baselines.py` (shared definitions and statuses; previously inline in the runner) |

### Final assumptions applied (D1–D7)

B = 1024; manifests and hashes authoritative (`SHA256SUMS`, provenance in every
output); GoogLeNet without auxiliary heads; synchronized BatchNorm; raw-input
gradient omitted; SGD momentum state; 1 GiB runtime reserve inside 12 GiB, exposed
in all runners (`--reserve`) and serialized; per-layer workspace 0. Nominal values:
`configs/nominal_experiment.json`.

### Runners, scripts and outputs

* `experiments/common.py` (shared arguments, provenance, atomic JSON writing);
  output schema `lhas-run-v2` with statuses, verification coverage and a runtime
  split (construction, search, baselines, verification; packing-cache state).
* `experiments/run_graph.py` and `run_gpipe.py` rewritten; `run_chain.py` updated.
* `scripts/final_queues.py` replaces `run_main_chain*.sh`, `run_sensitivity.sh`
  and the old queue files; `scripts/results_lib.py`, `make_figures.py` and
  `make_tables.py` read the v2 schema and carry statuses into tables and figures.
* Colab notebook and ingestion: see D17 in `docs/decision_table.md`.
* Provisional results of commit 757bd4c moved unchanged to
  `results/archive/provisional_757bd4c/`.

### After the Colab run (Tesla T4, 2026-09-28)

* The unmodified ZIP is stored in `results/colab/Tesla_T4_2026-09-28/` and ingested
  (accepted by every check). `scripts/analyse_profile.py` reproduces the notebook's
  held-out split and adds time-weighted, size-resolved and DP-scaling statistics and
  an affine-model comparison.
* Profiled input mode in the runners: `--profile` (measured T_cp; configurations
  whose local shape was not measured are excluded for every method) and `--red-rate`
  (β_red in byte/s). Queue `profiled` (N = 64, 256).
* Ingestion keeps the last row per shape after a resumed notebook run.

### Audit of commit 0892134 (2026-09-29)

Findings of an independent code audit (report not included). Regressions:
`tests/test_review_0892134.py` (F1 fixture and GoogLeNet case, F3, F4, F5, T1, T5, T6 chain
and graph), `tests/test_colab_safeguards.py` (notebook v3 resume guard, hardened ingestion,
stored T4 table reproduced exactly).

* `lhas.planner.graph_eval`: masked MP output operations keep `alts` and `spec` (F1).
* `experiments/run_gpipe.py`: not established when a search without incumbent has an
  unresolved All-reduce selection (F3); `--profile` rejected (T5).
* `lhas.baselines.common.metropolis`: one cached evaluator for the random-start search,
  the start evaluations and the proposals; `counts_initialization`, `counts_proposals`,
  `seconds_initialization`, `evaluated_plan_digests`; runners add
  `evaluations_sum_per_seed`, `evaluations_globally_distinct` (F4).
* Best-common DP: per-count lower bounds and `certification` for the branch engine (F5);
  `unavailable` status, `unavailable_counts` and `candidate_domain` when a profile lacks an
  entry, in both engines and for fixed and single-node baselines (T6).
* Provenance: `source_sha256`, `nominal_spec_sha256`, `uncommitted_diff_sha256`.
* T4 analysis rewritten (T1, T3, T4); superseded outputs in
  `results/processed/superseded_0892134/`. Notebook v3; ingestion requires all columns,
  finite ordered timings, per-use parent widths, unique signatures and a resume fingerprint
  for v3.
* Reruns: `docs/rerun_dependency_analysis.md`; superseded raw outputs in
  `results/raw/archive_0892134/`; `scripts/compare_reruns.py`.

### Audit of commit ac66b9e (2026-09-30)

Findings of an independent code audit (report not included). Regressions: `tests/test_review_ac66b9e.py`.

* `scripts/analyse_profile.py` (A1): `_config_times` follows the nominal raw-input-gradient
  policy; equal-FLOP pairs report their composition (p = 1 identical kernels, DP/MP faster).
* `scripts/ingest_colab.py` (A2): documented guarantee narrowed.
* `colab/build_notebook.py`, notebook v3 regenerated (A3): metadata written after the resume
  guard, first-session metadata kept, unfingerprinted data refused; `colab/cpu_smoke_test.py`
  runs first, compatible and rejected sessions.
* `scripts/compare_reruns.py` (A4): required pairs, seed identities, recorded incumbent-cost
  traces, nonzero exit on missing files.
* Docs: protocol, verification, REPORT, dependency analysis (A5 and A6 concerned manuscript
  wording only).

### Audit of commit e8f8d15 (2026-09-30)

* `colab/build_notebook.py` and regenerated notebook: reduction resume, stronger resume
  fingerprint (the audit's proposed patch, checked before adoption); `colab/cpu_smoke_test.py` covers reductions;
  `tests/test_final_resume_review.py`.
* README commands.
