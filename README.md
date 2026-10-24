# LHAS: Layer-wise Hybrid Parallelism for Distributed DNN Training

LHAS is a research implementation for studying distributed neural network training
on wavelength-routed optical interconnects. It selects data parallelism (DP) or
output-feature model parallelism (MP), together with a node count, for each layer
and evaluates these choices using optical-aware communication schedules.

This repository accompanies **“Layer-wise Hybrid Parallelism for Distributed DNN
Training on Optical Interconnects,”** currently under review at *ACM Transactions
on Modeling and Performance Evaluation of Computing Systems* (TOMPECS).
It contains the implementation, experiment configurations, tests, stored results,
plotting scripts, and a notebook for profiling local GPU computation. The
manuscript is not included in this repository.

**How to interpret the results:** distributed training times are model predictions.
The main evaluation uses a hypothetical P100 PCIe 16 GB computation reference.
The repository also includes real measurements of local computation on one Tesla
T4 and a separate sensitivity study that uses those measurements as computation
inputs. Neither study measures distributed execution on an optical system.

## Start here

| What you want to do | Where to start |
|---|---|
| Read the numerical results | [Result tables](results/processed/tables.md) and [paper figures](figures/presentation/) |
| Reproduce one small simulation | [Quick start](#quick-start) |
| Regenerate the evaluation figures and tables | [Figures and tables](#figures-and-tables) |
| Run or modify experiments | [Experiment configuration](#experiment-configuration) and [running-experiments](#running-experiments) |
| Understand the implementation and checks | [Model and verification](#model-and-verification) |
| Collect computation measurements on another GPU | [GPU profiling](#gpu-profiling) |

Browsing the stored results requires no installation. Simulation, analysis of the
stored T4 data, and figure generation run on a CPU. A GPU is needed only to collect
new GPU timing measurements.

## Quick start

The commands below use Linux/Bash and Python 3.11. The recorded experiment
environment used Python 3.11.15; its dependency versions are listed in
[environment/requirements-lock.txt](environment/requirements-lock.txt).

### 1. Set up the environment

```bash
git clone https://github.com/TravisDai/LHAS.git
cd LHAS

python3.11 -m venv .venv
source .venv/bin/activate

# Pinned dependencies for simulation, analysis, plots, and tests without PyTorch.
python -m pip install \
  numpy==2.4.4 numba==0.67.0 llvmlite==0.49.0 \
  pytest==9.1.1 matplotlib==3.10.9 nbformat==5.11.1
```

Run subsequent commands from the repository root. PyTorch and torchvision are
optional for this quick start; the [verification section](#model-and-verification)
explains when they are needed.

### 2. Reproduce a small case

This example evaluates AlexNet on a modeled 64-node ring. It compares LHAS with
DP on all nodes, the best feasible common-count DP, and adapted One-Weird-Trick
(OWT). The output goes to a separate directory, preserving the supplied results.

```bash
python experiments/run_chain.py --workload alexnet --N 64 \
  --baselines dp,dpbest,owt --out results/local --tag quickcheck

python scripts/check_reproduction.py \
  results/local/quickcheck/alexnet_N64.json
```

The comparison should finish with `RESULT: agrees`. It checks the LHAS plan,
predicted cost and status; the included baseline costs and statuses; and the
source, workload, and specification hashes against
[the stored AlexNet result](results/raw/final/alexnet_N64.json). It also reports
which baselines in the stored result were not rerun.

This is a small reproduction check. Full experiment queues, particularly at large
node counts, require substantially more time and memory.

## Figures and tables

The stored raw results are sufficient to regenerate the evaluation figures and
tables without rerunning the experiments or collecting new GPU measurements.

```bash
# Paper Figures 6–10 and 12, plus a snapshot of the plotted inputs.
python scripts/make_presentation_figures.py --output figures/presentation

# Grouped T4 computation-model analysis, its report, and Figure 11.
python scripts/analyse_profile.py

# Refresh the summary CSV files and additional artifact plots first.
python scripts/make_figures.py

# Then generate the numerical tables from those summaries and raw results.
python scripts/make_tables.py
```

These commands refresh generated files in `figures/` and `results/processed/`.
The presentation figures are exported as PDF, PNG, and SVG. Their input snapshot
records the SHA-256 hash of every raw result used.

<details>
<summary>Figure and table mapping</summary>

Numbering follows the revised manuscript associated with this artifact.

| Paper item | Content | Generated output |
|---|---|---|
| Figure 6; Table 7 | Main comparison | `fig_main_comparison_compact.*`; main comparison in `tables.md` |
| Figure 7; Tables 8–9 | Planner ablations and collective families | `fig_ablations_families.*`; corresponding sections in `tables.md` |
| Figure 8; Table 10 | One-at-a-time sensitivity | `fig_sensitivity_compact.*`; sensitivity section in `tables.md` |
| Figure 9; Table 11 | Batch-size sensitivity | `fig_batch_comparison.*`; batch section in `tables.md` |
| Figure 10; Table 12 | Selected layer configurations | `fig_layer_configurations.*`; configuration listings in `tables.md` |
| Table 13 | Verification coverage and planner runtime | Coverage section in `tables.md`; runtime fields in `main_summary.csv` |
| Figure 11; Table 14 | T4 computation-model validation | `figures/fig_t4_compute_validation.*`; `results/processed/t4_validation.json` and `.md` |
| Figure 12; Table 15 | Measured-computation sensitivity | `fig_profiled_comparison.*`; measured-computation section in `tables.md` |

Unless a full path is shown, figure files are in `figures/presentation/` and
tables and summaries are in `results/processed/`.

The four-node worked example in Table 2 is checked by:

```bash
python -m pytest -q tests/test_worked_example.py
```

Figures 1–5 and Tables 1 and 3–6 describe the architecture, method, definitions,
or notation; they are not generated from experiment outputs.

See [the figure reproduction guide](docs/presentation_figures.md) for input
directories and snapshot checks. Other plots in `figures/` provide additional
views of the stored results.

</details>

## Experiment configuration

Defaults come from
[configs/nominal_experiment.json](configs/nominal_experiment.json), with optical
and computation inputs in
[configs/approved_reference_parameters.json](configs/approved_reference_parameters.json).
Command-line options override these values, and each result records the settings
used. The [model specification](docs/decision_table.md) explains the choices in
detail.

| Setting | Nominal value |
|---|---|
| Workloads | AlexNet, VGG16, GoogLeNet, and ResNet50; GoogLeNet has no auxiliary heads |
| Workload definitions | Pinned graph manifests in `configs/workloads/`, exported with torchvision 0.29.0; hashes in `SHA256SUMS` |
| Global minibatch | 1,024, fixed across node counts; a separate sensitivity uses 256 at 64 nodes |
| Modeled system sizes | 64, 128, 256, and 512 nodes; AlexNet at 1,024 nodes is supplementary |
| Computation | Analytical FLOPs with 50% utilization of a hypothetical P100 PCIe 16 GB reference |
| Precision | float32 |
| Optical inputs | 64 wavelength labels per direction; 25 Gbit/s per wavelength; 25 μs circuit setup |
| Compute–communication overlap | None in the nominal layer-wise objective; varied in sensitivity runs |
| BatchNorm | Synchronized full-batch statistics; DP charges separate forward and backward statistics reductions, while MP keeps statistics channel-local |
| Gradient of the raw input | Omitted; internal input gradients are retained |
| Optimizer storage | SGD momentum with one parameter-sized state buffer |
| Memory budget | 12 GiB per node, including a 1 GiB runtime reserve; per-layer workspace is set to zero |

These are modeling inputs. In particular, the memory ledger uses accumulated
reservations rather than a measured peak GPU allocation.

## Running experiments

Use `run_chain.py` for AlexNet and VGG16, and `run_graph.py` for the supported
GoogLeNet and ResNet50 graphs. Adapted GPipe is evaluated only on the chain
workloads.

```bash
# Inspect the available options.
python experiments/run_chain.py --help

# Run an analytical experiment with the default baselines.
python experiments/run_chain.py --workload alexnet --N 64 --tag example

# Run a branch workload.
python experiments/run_graph.py --workload resnet50 --N 128 --tag example

# Evaluate adapted GPipe on a chain workload.
python experiments/run_gpipe.py --workload vgg16 --N 256 --tag example_gpipe
```

Results are written to `results/raw/<tag>/<workload>_N<N>.json`. Use distinct tags
or `--out` directories for your own runs.

The comparison methods include DP on all nodes, best common-count DP, adapted
OWT, a FlexFlow-derived Metropolis search, and adapted GPipe. These are
implementations of the stated baseline contracts under the shared cost model;
they are not executions of the original distributed-training frameworks.
See [the specification](docs/decision_table.md) and
[experiment guide](docs/experiments.md) for their definitions and search budgets.

<details>
<summary>Run the full experiment queues</summary>

The queue generator writes the command lists for the reported experiment set.
These commands target `results/raw/final*`, so use a separate checkout when
rerunning them to keep the supplied reference outputs intact.

```bash
python scripts/final_queues.py
mkdir -p results/logs results/cache/packings
export LHAS_PACK_CACHE_DIR="$PWD/results/cache/packings"

bash scripts/run_list.sh results/queues/chain_main.txt chain_main
bash scripts/run_list.sh results/queues/graph_main.txt graph_main
bash scripts/run_list.sh results/queues/gpipe.txt gpipe
bash scripts/run_list.sh results/queues/ablations.txt ablations
bash scripts/run_list.sh results/queues/family.txt family
bash scripts/run_list.sh results/queues/batch.txt batch
bash scripts/run_list.sh results/queues/sens_chain.txt sens_chain
bash scripts/run_list.sh results/queues/sens_graph.txt sens_graph
bash scripts/run_list.sh results/queues/profiled.txt profiled

# Optional construction-runtime and supplementary large-system runs.
bash scripts/run_list.sh results/queues/cold.txt cold
bash scripts/run_list.sh results/queues/supp.txt supp
```

The profiled queue uses the included T4 input table; it does not collect new
measurements. The supplementary queue contains AlexNet at 1,024 nodes.
Corresponding 1,024-node results for the other three workloads are not included.

Schedule construction becomes expensive at large node counts. The packing cache
reuses eligible route packings with matching transport inputs, including when an
interrupted experiment is restarted. Each queue writes command output and exit
codes to `results/logs/<name>.log`; inspect these logs for failed commands.

</details>

### Reading result statuses

Each JSON result records the model settings, selected configurations and
collective candidates, cost and memory breakdowns, search statistics, verification
coverage, execution times, and provenance.

| Status | Meaning |
|---|---|
| `feasible` | A plan satisfies the modeled constraints. Optimality depends on the method and any recorded certification. |
| `incumbent` | The best completed feasible result is retained, but optimality has not been established. |
| `infeasible` | Infeasibility has been established within the stated model and candidate domain. |
| `not_established` | The evaluation did not establish a result; this is not proof of infeasibility. |
| `unavailable` | A required measured computation input is missing. |

For best common-count DP, also inspect `certification` and `candidate_domain`.
For Metropolis searches, inspect the unresolved-proposal counts and search budget.
A feasible result alone does not certify global optimality.

## Model and verification

The implementation includes deterministic routing and wavelength packing,
adaptations of OSM, WRHT, and mixed-radix OpTree, unequal-group communication
candidates, and exchanges at layer and branch boundaries. An independent
verifier in `src/lhas_verify/` checks exported communication schedules without
importing the schedule-construction implementation.

The configuration searches have different guarantees:

- **Chain networks:** memory-aware dynamic programming minimizes the modeled
  objective over the represented feasible candidates.
- **Supported branch networks:** the search minimizes an additive objective and
  then checks memory. A memory-feasible minimizer is certified within that
  candidate family; a general memory-constrained branch search is not implemented.
  The objective does not simulate optimal concurrent execution on shared GPUs.

Run the tests with skip reasons visible:

```bash
python -m pytest -q -rs tests
```

The suite includes tensor-ownership checks, comparison with exhaustive search on
small instances, invalid-schedule rejection, and regression tests. With the
CPU-only dependency set above, tests that require PyTorch or torchvision are
skipped.

For graph export, the PyTorch BatchNorm equivalence test, and the notebook smoke
tests, install the full recorded environment:

```bash
python -m pip install -r environment/requirements-lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130

python -m pytest -q -rs tests
python colab/cpu_smoke_test.py
```

The smoke test exercises notebook logic on CPU; it does not collect GPU timings.
Existing workload manifests can be checked independently with:

```bash
(cd configs/workloads && sha256sum -c SHA256SUMS)
```

Verification records distinguish checks at actual payloads, checks at reduced
payloads, and operations skipped because of resource limits. Recorded plan
verification covers selected LHAS operations; baseline plans were not
individually rechecked, and adapted GPipe has no recorded schedule check.
Implementation verification does not establish optical hardware accuracy.

See [verification scope and coverage](docs/verification.md),
[the memory ledger](docs/memory_ledger.md), and
[current limitations](docs/STATUS.md).

## GPU profiling

The [profiling notebook](colab/LHAS_compute_profiling.ipynb) measures local forward,
weight-gradient, and input-gradient products at the DP and MP shapes used by
the planner.

1. Open the notebook in Google Colab.
2. Select a GPU runtime, then choose **Run all**.
3. Download the generated `lhas_profile_<gpu>.zip` and retain it unchanged as the
   raw measurement archive.
4. Convert it to a planner input table:

```bash
python scripts/ingest_colab.py path/to/lhas_profile_Tesla_T4.zip \
  --out results/local/profiles
```

Replace the example ZIP path with the downloaded file. Ingestion checks the
workload manifests, batch size, tensor shapes, precision settings, and timing
records. Unmeasured configurations remain unavailable to every planner in
profiled mode; they are not filled using scaled timings.

The stored Tesla T4 measurements are in
[results/colab/Tesla_T4_2026-09-28/](results/colab/Tesla_T4_2026-09-28/), with their
ingested table in
[configs/profiles/profiled_Tesla_T4_B1024.json](configs/profiles/profiled_Tesla_T4_B1024.json).
They were collected using notebook **version 2**. The current notebook is
**version 3**, with additional resume safeguards and CPU smoke checks; no GPU
measurements from version 3 are included.

The following example uses the stored T4 table and its model-equivalent local
reduction rate:

```bash
python experiments/run_chain.py --workload alexnet --N 64 \
  --profile configs/profiles/profiled_Tesla_T4_B1024.json \
  --red-rate 218.6e9 --baselines dp,dpbest,owt \
  --out results/local --tag profiled_T4
```

The `--red-rate` value is in bytes per second. Use a rate derived for the measured
device when supplying a different profile. GPipe does not support profiled input
mode. The included `scripts/analyse_profile.py` analyzes the stored T4 run;
the [grouped validation protocol](docs/t4_validation_protocol.md) describes its
calibration and held-out split by kernel signature.

## Repository guide

| Path | Contents |
|---|---|
| `configs/` | Optical and computation inputs, nominal settings, workload manifests, and measured computation tables |
| `src/lhas/transport/`, `src/lhas/collectives/` | Routing, wavelength packing, timing, and collective candidates |
| `src/lhas/model/`, `src/lhas/planner/` | Tensor layouts, cost and memory accounting, and configuration search |
| `src/lhas/baselines/` | Baseline definitions and implementations |
| `src/lhas_verify/`, `src/lhas/verify_plan.py` | Independent schedule verifier and selected-plan coverage checks |
| `experiments/`, `scripts/` | Experiment runners, queues, analysis, plotting, and reproduction checks |
| `colab/` | GPU profiling notebook, notebook generator, and CPU smoke test |
| `results/raw/final*/` | Retained evaluation outputs |
| `results/processed/`, `figures/` | Summaries, numerical tables, and generated figures |
| `results/colab/` | Raw GPU profiling archives |
| `tests/`, `docs/` | Tests, specification, verification documentation, and implementation notes |

## Provenance and reuse

Raw outputs record the configurations and development commits that produced
them. [PROVENANCE.md](PROVENANCE.md) explains how these records relate to the
public repository, including source digests and the retained supplementary runs.
Superseded outputs are kept in archive directories for traceability and are not
part of the current evaluation.

The [implementation report](docs/REPORT.md) records completed checks, while
[docs/STATUS.md](docs/STATUS.md) lists remaining limitations.
[docs/AI_USE.md](docs/AI_USE.md) describes the use of generative AI tools.

When reporting results obtained with this artifact, cite the accompanying paper
and record the repository commit and configuration used. You can obtain the
commit with `git rev-parse HEAD`. For questions or reproducibility problems,
[open an issue](https://github.com/TravisDai/LHAS/issues) with the command,
environment, commit, and relevant output or log.

## License

Released under the [MIT License](LICENSE).
