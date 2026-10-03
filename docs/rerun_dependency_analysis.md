# Rerun dependency analysis for the findings of LHAS_0892134_Review.md

Date: 2026-09-29. Base: commit 0892134. Fix commits: 71465c6 (F1, F3, F4, F5, T5, T6),
4cff71f (T1, T3, T4; notebook v3; ingestion hardening). Superseded outputs are archived
unchanged in `results/raw/archive_0892134/` and `results/processed/superseded_0892134/`.

## What each fix can change

| Finding | Code change | Outputs that can change | Outputs that cannot |
|---|---|---|---|
| F1 | graph evaluator keeps schedule alternatives and verification specifications of masked MP output operations | status of graph fixed-plan evaluations that fail the budget with a non-final MP layer (infeasible becomes not established); graph Metropolis counters; verification of such operations | costs; feasibility of any plan (the min-charge combination and its reservation are unchanged); recorded Metropolis incumbent-cost traces (random numbers are drawn only for feasible proposals); chain outputs |
| F3 | GPipe without incumbent and with unresolved All-reduce selection is not established | GPipe runs with no feasible plan | every stored GPipe run (all feasible, `allreduce_selection_unresolved = 0`) |
| F4 | random-start search and start evaluations counted (initialization phase); per-seed and global distinct counts; initialization time | every Metropolis output (counts, seconds, trace x-axis) | best costs and recorded incumbent-cost traces (same random streams, same cached evaluator) |
| F5 | branch best-common DP serializes per-count lower bounds and certification | branch `dp_best` records with unresolved counts | stored branch `dp_best` records have no unresolved counts; chain records already carried bounds |
| T5 | `run_gpipe.py --profile` rejected | none stored (no profiled GPipe run) | all |
| T6 | missing profile entry is `unavailable`, not infeasible; best-common DP enumerates all divisor counts and lists unavailable ones | profiled runs: VGG16 `dp_best.per_count` labeled count 4 infeasible instead of unavailable and omitted counts 1 and 2 (8 and 16 are memory infeasible) | best-common DP value and certificate (p = 32 excluded by its lower bound; reviewer check confirmed) |
| T1, T3, T4 | grouped validation by timed kernel signature; fuller metrics; scaling reference | `results/processed/t4_validation.*`, `figures/fig_t4_compute_validation.*` | the T4 ZIP and the ingested compute table (fresh ingestion reproduces it exactly; regression test) |
| provenance | `source_sha256`, `nominal_spec_sha256`, hash of any uncommitted diff | new outputs only | — |

## Decisions

* **Rerun (clean commit, `LHAS_PACK_CACHE_DIR=results/cache/packings`)**: the 16 primary runs
  (`results/raw/final/`), the supplementary AlexNet N = 1024 run (`final_supp/`) and the 8
  profiled runs (`final_profiled_T4/`). Reasons: F4 affects every Metropolis output, F1 the
  graph Metropolis counters, T6 the profiled VGG16 per-count statuses, and three primary runs
  had uncommitted source (`code_dirty`: AlexNet N = 512, ResNet50 N = 64, VGG16 N = 256). A
  full rerun gives one clean provenance for all primary results and is an end-to-end check
  that the fixes leave plan costs unchanged. Queues: `results/queues/rerun_A.txt`,
  `rerun_B.txt`; logs `results/logs/rerun_{A,B}.log`.
* **Remain valid, not rerun**: `final_gpipe/`, `final_supp_gpipe/` (F3 inapplicable, T5 not
  used), `final_sens/`, `final_family/`, `final_batch/`, `final_cold/`, `final_ablations/` (no
  Metropolis search; no branch `dp_best` with unresolved counts; no profile). The new
  `certification` and `unavailable_counts` fields are absent from these files; readers treat
  them as optional.
* **Regenerated from raw outputs**: `results/processed/*` tables and summaries and all figures
  (`scripts/make_figures.py`, `scripts/make_tables.py`, `scripts/analyse_profile.py`).

## Checks after the rerun

`scripts/compare_reruns.py` compares every rerun output with its archived predecessor: LHAS
cost, configurations and status; every baseline cost and status; each Metropolis seed's
best cost, best plan and accepted count; and reports the changed status and count fields.
Since review ac66b9e (A4) it also requires all 25 archived/rerun pairs to exist, matches
Metropolis runs by seed (missing or duplicated seeds fail), and compares each seed's
recorded incumbent-cost trace (iteration and best cost so far; evaluation counts and
elapsed times may differ). It exits nonzero on any difference or missing file. The evidence
is therefore equality of recorded incumbent-cost traces, not of complete search-state
trajectories, which are not recorded. Result: 25 of 25 pairs, 51 traces, 0 differences.
Its output is `results/processed/rerun_comparison.md`.
