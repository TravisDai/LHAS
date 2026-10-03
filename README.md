# LHAS reproducibility artifact

A reimplementation of the approved analytical model of the paper "LHAS: layer-wise
hybrid parallelism on wavelength-routed optical interconnects". The paper is not yet
published, and its manuscript is not part of this repository. The artifact covers:

* a physical transport scheduler with deterministic first-fit;
* collective candidate families: adapted OSM, WRHT, mixed-radix OpTree, and unequal-group A/B/C;
* boundary exchanges and typed fork/join interfaces;
* analytical computation and a conservative accumulated-reservation memory ledger;
* memory-aware configuration search.

It also includes an **independent schedule verifier**, the baselines (pure DP,
adapted OWT, a FlexFlow-derived Metropolis search, an optical-adapted GPipe),
experiment scripts, figures, and a Colab notebook that profiles local GPU
computation.

> **Status.** All times produced here are predictions of an analytical model:
> the hypothetical P100 PCIe 16 GB reference with the approved optical parameters.
> None of them is a measurement. The original profiles and scripts are lost, and the
> historical results are not reconstruction targets.
>
> This repository is a **public checkpoint** of the verified implementation and evaluation
> results. It was prepared from a private development repository; the experiments were run
> from development commits whose hashes are recorded in the raw outputs. `PROVENANCE.md`
> maps the public tree to those commits (the Python source is byte-identical to the source
> that produced the primary, supplementary and profiled results). Outstanding work and open
> audit items: `docs/STATUS.md`. No license and no archival release yet.

## Final specification

The experimental specification (decisions D1–D17, `docs/decision_table.md`) is
fixed in `configs/nominal_experiment.json` and used as the default by every runner:

| Item | Nominal value |
|---|---|
| global batch | B = 1024 (sensitivity: B = 256 at N = 64) |
| system sizes | N = 64, 128, 256, 512 (AlexNet N = 1024 supplementary) |
| workloads | manifests `configs/workloads/*.json` (sha256 in `SHA256SUMS`); GoogLeNet without auxiliary heads |
| BatchNorm | synchronized full-batch statistics (DP: 2C forward and 2C backward reductions; MP: none) |
| raw-input gradient | omitted (nontrainable graph input); `--raw retain` is a sensitivity |
| optimizer state | SGD momentum, one parameter-sized buffer (`--opt-state 1`) |
| memory budget | 12 GiB per node, of which a 1 GiB runtime reserve is charged once (`--reserve`) |
| per-layer workspace | 0 |

## Layout

```
configs/approved_reference_parameters.json   frozen optical + compute inputs (author-approved)
configs/nominal_experiment.json               final experimental specification (D1-D17)
configs/workloads/*.json, SHA256SUMS          pinned torchvision 0.29.0 graph manifests (shapes only) and hashes
environment/requirements-lock.txt             pinned simulator environment
src/lhas/params.py                            parameter loading and validation
src/lhas/transport/                           routing, first-fit packing (numba), timing, setup reuse, export
src/lhas/collectives/                         OSM, WRHT, OpTree, unequal-group families; alternatives service
src/lhas/model/                               workloads, layer reservations, boundary matrices, chain and typed-interface charges, profile safeguards
src/lhas/planner/                             memory-aware chain DP; typed-interface exact search
src/lhas/baselines/                           common definitions and statuses; chain and branch baselines; GPipe
src/lhas/verify_plan.py                       re-verifies every selected operation with coverage records
src/lhas_verify/                              independent verifier (imports nothing from lhas)
experiments/run_chain.py, run_graph.py        one workload x one system size -> results/raw/<tag>/*.json
experiments/run_gpipe.py                      optical-adapted GPipe (chain workloads)
scripts/                                      queues, figures, tables, graph export, pilot, Colab ingestion, diagnostics
colab/                                        profiling notebook (+ generator and CPU smoke test)
docs/                                         specification, memory ledger, verification, change log, experiments,
                                              report, status of outstanding work
results/raw/final*                            final results; results/raw/archive_0892134/ and results/archive/ hold
                                              superseded outputs (kept unchanged for traceability)
results/processed/, figures/                  summaries, tables and figures generated from the raw results
results/colab/                                the author's unmodified Tesla T4 profiling ZIP (notebook v2)
PROVENANCE.md                                 relation between this public tree and the development commits
tests/                                        pytest suite
```

## Environment

Python 3.11.15 with the versions in `environment/requirements-lock.txt` (numpy
2.4.4, numba 0.67.0, pytest, matplotlib; torch 2.14.0 and torchvision 0.29.0 only
for re-exporting manifests, the BatchNorm equivalence test and the notebook smoke
test). This package was developed on a 2-core, 8 GB Linux container.

```bash
pip install -r environment/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

## Exact commands

```bash
# 1. tests: verifier, rejection of invalid schedules, ownership, exhaustive DP,
#    engine cross-check, audit regressions, notebook and ingestion safeguards
python -m pytest -q tests

# 2. (optional) re-export the pinned graph manifests and check their hashes
python scripts/export_graphs.py && (cd configs/workloads && sha256sum -c SHA256SUMS)

# 3. write the final command lists (results/queues/*.txt) and run them
python scripts/final_queues.py
export LHAS_PACK_CACHE_DIR=$PWD/results/cache/packings
scripts/run_list.sh results/queues/chain_main.txt chain_main   # AlexNet, VGG16, N = 64..512
scripts/run_list.sh results/queues/graph_main.txt graph_main   # GoogLeNet, ResNet-50, N = 64..512
scripts/run_list.sh results/queues/gpipe.txt gpipe
scripts/run_list.sh results/queues/ablations.txt ablations
scripts/run_list.sh results/queues/family.txt family
scripts/run_list.sh results/queues/batch.txt batch
scripts/run_list.sh results/queues/sens_chain.txt sens_chain
scripts/run_list.sh results/queues/sens_graph.txt sens_graph
scripts/run_list.sh results/queues/cold.txt cold               # empty packing cache (runtime study)
scripts/run_list.sh results/queues/supp.txt supp               # AlexNet N = 1024 (supplementary)
scripts/run_list.sh results/queues/profiled.txt profiled       # measured T4 computation, N = 64, 256

# single runs, for example
python experiments/run_chain.py --workload alexnet --N 64 --tag final
python experiments/run_graph.py --workload resnet50 --N 128 --tag final
python experiments/run_gpipe.py --workload vgg16 --N 256 --tag final_gpipe

# 4. construction-cost pilot (one all-pairs first-fit phase)
python scripts/pilot_allpairs.py 1024:256 1024:512 1024:1024

# 5. T4 compute-model analysis; summaries, tables and figures; comparison with the archived
#    provisional results
python scripts/analyse_profile.py
cd scripts && python make_figures.py && python make_tables.py && python compare_provisional.py && cd ..

# 6. check reruns against the archived outputs of commit 0892134 (nonzero exit on any
#    difference in costs, plans or recorded incumbent-cost traces, or a missing file)
python scripts/compare_reruns.py
```

**Cheap reproduction check** (under a minute: 19 s on a 2-core container; no queue, no GPU). It reruns one
primary configuration into a scratch directory and compares it with the stored result:

```bash
python experiments/run_chain.py --workload alexnet --N 64 --baselines dp,dpbest,owt \
  --out /tmp/lhas_check --tag check
python scripts/check_reproduction.py /tmp/lhas_check/check/alexnet_N64.json
```

The second command compares the LHAS cost, selected configurations and status, and the DP(N),
best-common-DP and OWT values, with `results/raw/final/alexnet_N64.json`, and checks that the
Python source digest equals the one recorded in the stored result.

**Large N.** Construction is exact and deterministic; its cost dominates at
N ≥ 512. `LHAS_PACK_CACHE_DIR` stores every packing with ≥ 50 000 routes, keyed by
the transport parameters and the route set, so interrupted runs resume without
repacking. `scripts/run_list.sh` records each command's exit code, so an
out-of-memory termination appears as `EXIT 137` rather than as a missing file.

Every run writes a JSON file (schema `lhas-run-v2`) containing the settings,
transport and compute parameters (including the runtime reserve), provenance
(manifest hash, code commit, packing-cache state at start), the selected
configurations and schedule candidates, the charge breakdown and reservation
summary, the status of every planner and baseline (`feasible`, `infeasible`,
`not_established`, `incumbent`, and `unavailable` for a missing measured compute input),
search statistics, verification coverage records,
and runtimes split into construction, search, baselines and verification.

## Colab profiling (to be run by the author)

1. Open `colab/LHAS_compute_profiling.ipynb` in Google Colab.
2. Runtime → Change runtime type → any GPU.
3. Runtime → Run all.
4. The last cell downloads `lhas_profile_<gpu>.zip`. Return it unchanged.
5. `python scripts/ingest_colab.py lhas_profile_<gpu>.zip` checks and converts it.

The notebook embeds the repository manifests and their hashes and stops before
profiling if the Colab torchvision graphs differ. It records the GPU, driver,
CUDA, cuDNN, PyTorch and torchvision versions and the precision settings (float32,
TF32 off). It times forward, input-gradient and weight-gradient products separately
at the actual local DP and MP shapes for B = 1024, with warm-up and CUDA-event
synchronization, keeps all raw samples, saves after every shape (a rerun in the
same runtime resumes), and releases every allocation in `finally` blocks. Shapes it
cannot run are recorded as unmeasured; it never divides full-layer times by p. It
reports a model-equivalent rate for the local reduction term, and fits a
constant-throughput model of *that* GPU on calibration shapes, validated on held-out
shapes (signed and absolute relative errors, tails, rank correlation; no statistics
below the declared minimum sample counts). The ingestion rejects any ZIP whose
batch, manifest hashes, precision or local shapes do not match. Results must not be
relabeled or rescaled as P100. `python colab/cpu_smoke_test.py` checks the notebook
logic and the ingestion checks on CPU only; it performs no GPU timing.

The author's Tesla T4 run is stored unmodified in `results/colab/Tesla_T4_2026-09-28/`
and ingested as `configs/profiles/profiled_Tesla_T4_B1024.json`;
`python scripts/analyse_profile.py` applies the grouped protocol of
`docs/t4_validation_protocol.md` (calibration and validation split by timed kernel
signature) and writes `results/processed/t4_validation.{json,md}` and
`figures/fig_t4_compute_validation.pdf`; the report and the figure are generated from the
JSON only. The T4 data were produced by notebook version 2. Version 3 (for future runs; not
yet run on a GPU) records one row per timed kernel signature, refuses to resume under a
changed configuration, device class, driver, CUDA, framework or precision policy, never
overwrites existing files on a rejected resume, and does not repeat completed reduction
measurements. A separately labeled planning run with measured
computation uses `--profile <table> --red-rate <byte/s>` (configurations without a
measured local shape are excluded for every method), for example
`python experiments/run_chain.py --workload alexnet --N 64 --profile configs/profiles/profiled_Tesla_T4_B1024.json --red-rate 218.6e9 --baselines dp,dpbest,owt --tag final_profiled_T4`.

## Documents

* `docs/decision_table.md`: final D1–D17 specification and where each item is implemented.
* `docs/memory_ledger.md`: per-operation reservation ledger and corrected aliasing rules.
* `docs/verification.md`: verification categories, coverage records, regression tests.
* `docs/change_log.md`: model equations → modules and tests; changes and audit fixes.
* `docs/experiments.md`: experiments, output directories and queues.
* `docs/rerun_dependency_analysis.md`, `docs/t4_validation_protocol.md`: rerun decisions and
  the T4 validation protocol.
* `docs/REPORT.md`: what was executed, passed, failed and remains unverified.
* `docs/STATUS.md`: outstanding work and open items.
* `PROVENANCE.md`: development commits, source digests and what differs in this public tree.

Identifiers such as 0892134, ac66b9e and e8f8d15 in test names, documents and generated reports
refer to development commits that were audited independently; "reviewer" in those places means
the code auditor. The audit reports are not included.

## License

No license has been chosen yet; the author will add one. This repository therefore contains
no license file at present.
