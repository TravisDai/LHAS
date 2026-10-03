# Experiments (final specification)

All planners and baselines share the same analytical compute inputs, global batch,
collective transport model and candidate families, memory accounting (including
SGD momentum state and the 1 GiB runtime reserve), and precision. "Predicted time"
is a model output; nothing here is measured on optical hardware. Nominal values:
`configs/nominal_experiment.json`. Command lists: `python scripts/final_queues.py`
writes `results/queues/*.txt`; run each with `scripts/run_list.sh <file> <log>`.

| ID | Experiment | Output tag (results/raw/…) | Queue |
|---|---|---|---|
| E1 | Main comparison, N ∈ {64, 128, 256, 512}, B = 1024. Methods: LHAS; DP (p = N); DP (best common p); OWT (adapted, D12); FlexFlow-derived Metropolis search (adapted, D14; 3 seeds × 1000 iterations, multi-start); single-node reference. Every selected LHAS operation is re-verified with coverage records. | `final/` | `chain_main`, `graph_main` |
| E1s | AlexNet N = 1024 (supplementary, D16) | `final_supp/`, `final_supp_gpipe/` | `supp` |
| E1g | Optical-adapted GPipe, AlexNet and VGG16, N = 64–512 (D13) | `final_gpipe/` | `gpipe` |
| E2 | Planner ablations: strategy only (fixed counts), DP only with per-layer counts, MP only with per-layer counts, greedy layer-wise (corrected) | `final_ablations/` (N = 64, 256) | `ablations` |
| E3 | Overlap sensitivity α ∈ {0, 0.25, 0.5, 0.75, 1} | `final_sens/alpha_*` (N = 64) | `sens_chain`, `sens_graph` |
| E4 | One-at-a-time sensitivity: ρ, reduction efficiency, setup delay, λ = W, reach (8, 15, 16, unrestricted without peer cap), ε, whole-item forwarding, raw-input gradient retained (D5), no runtime reserve (D7), no momentum state (D6) | `final_sens/*` (N = 64) | `sens_chain`, `sens_graph` |
| E5 | Collective-family comparison replacing LHAS1–3 (D15): all vs one-stage vs deepest, LHAS re-optimized within each family, retained candidate IDs exported | `final_family/{all,one-stage,deepest}/` (N = 64) | `family` |
| E6 | Batch sensitivity B = 256 at N = 64: LHAS, DP(N), best-common DP, OWT (D1) | `final_batch/B256/` | `batch` |
| E7 | Planner runtime: construction vs configuration search, reused vs cold packing cache (D16) | `runtime` fields of every output; `final_cold/` | `cold` |
| E8 | Implementation verification: independent verifier, exhaustive checks, engine cross-check, regression tests of the defects found in code audits | `pytest` | — |
| E9 | Compute-model validation on the actual Colab GPU (held-out kernel signatures, grouped split; D17), run by the author on a Tesla T4 | `results/colab/Tesla_T4_2026-09-28/`; `scripts/ingest_colab.py`; `scripts/analyse_profile.py` | — |
| E10 | Profiled planning sensitivity: measured T4 computation and reduction rate, N = 64 and 256; LHAS, DP(N), best-common DP, OWT | `final_profiled_T4/` | `profiled` |

Evidence categories are kept separate: E8 is implementation verification; E9 is a
compute measurement on one GPU; no experiment validates the optical model against
hardware.

Old results produced at commit 757bd4c under provisional settings are archived
unchanged in `results/archive/provisional_757bd4c/` and are not part of these
experiments. The packing cache `results/cache/packings` was reused: its entries are
keyed by the transport inputs (N, λ, W, reach, peer cap, κ and the route set), which
the final specification does not change.

Results reported for earlier versions of this work (speedups, collective sensitivity and
FC node counts) are not targets. No parameter was tuned toward them.
