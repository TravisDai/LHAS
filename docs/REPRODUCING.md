# Reproducing the LHAS study

This guide separates inspecting stored evidence, checking a small run, regenerating
reports, and rerunning the full experiment set. The nominal specification is
[`configs/nominal_experiment.json`](../configs/nominal_experiment.json);
[experiments.md](experiments.md) maps each study to its outputs.

## 1. Install the CPU environment

The recorded reference environment is Python 3.11.15 on Linux. Use Python 3.11
with the pinned dependencies below. A GPU is not needed for simulation or for
analyzing the stored Tesla T4 measurements.

```bash
git clone https://github.com/TravisDai/LHAS.git
cd LHAS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-cpu.txt
```

Run subsequent commands from the repository root. The CPU requirements are an
exact subset of the full reference requirements, excluding torch and torchvision.
The experiment runners and tests configure their own source import paths. For
interactive imports, use `export PYTHONPATH="$PWD/src"`.

The implementation was developed on a two-core Linux container with 8 GB RAM.
This is a reference machine, not a sufficient resource guarantee for every
experiment: constructing large schedules can be expensive, especially at
512 and 1,024 nodes. Runtime and peak-resident-memory records are stored with
the results. There is no need to rerun the full queues to inspect the study.

## 2. Check one stored result

```bash
LHAS_CHECK_DIR=$(mktemp -d)
python experiments/run_chain.py --workload alexnet --N 64 \
  --baselines dp,dpbest,owt --out "$LHAS_CHECK_DIR" --tag check
python scripts/check_reproduction.py "$LHAS_CHECK_DIR/check/alexnet_N64.json"
```

Expected final output: `RESULT: agrees`. Costs are compared with a relative
tolerance of `1e-12`; source, workload, and nominal-specification hashes must
also match. This check covers LHAS and the requested DP/OWT baselines. It does
not rerun Metropolis or GPipe. First-use compilation affects elapsed time.

```bash
python -m pytest -q -rs tests
```

With the CPU requirements, PyTorch-dependent checks are skipped: the two notebook
smoke-test integrations and the three synchronized-BatchNorm cases. The suite also
contains an intentional skip where transport participants exceed the ring size.
Use the reported reasons, rather than a total pass count alone, to describe coverage.

## 3. Inspect and regenerate reports

The stored [tables](../results/processed/tables.md),
[figures](../figures/), and [T4 report](../results/processed/t4_validation.md)
can be read without installing anything. To regenerate them from the recorded
inputs, run:

```bash
python scripts/analyse_profile.py
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/compare_reruns.py
```

These commands rewrite generated reports and figures in this checkout. They do
not rerun the experiment queues or measure a GPU. Use a separate checkout if you
want to preserve the original generated files for comparison. The profile analysis
writes `results/processed/t4_validation.json`, its Markdown report, and the T4
figure. The other reporting scripts consume `results/raw/final*`; their tables
retain feasibility and resource-limit statuses. The rerun comparison checks 25
stored output pairs and their recorded incumbent-cost traces, not full search
trajectories. `python scripts/compare_provisional.py` optionally compares the
older provisional settings with the final study.

## 4. Understand statuses and verification

| Status | Interpretation |
|---|---|
| `feasible` | A plan satisfying the modeled reservation budget was established; consult the method's certification for its optimization guarantee |
| `infeasible` | Infeasibility was established within the declared domain, by exact evaluation or a valid reservation lower bound |
| `not_established` | The available evaluation did not establish feasibility or an optimum; it is not an infeasibility proof |
| `incumbent` | Best completed result when unresolved candidates prevent certification |
| `unavailable` | A required measured local-computation input is absent; that configuration is excluded from the profiled domain |

The independent verifier records `checked-actual`, `checked-reduced`, `skipped`,
or `local` for each selected LHAS operation. A reduced-payload check or a check
without re-deriving first-fit packing has narrower coverage than a full check.
Baseline plans were not all individually verified. See
[verification.md](verification.md) for thresholds and the evidence each check supplies.

## 5. Full environment and optional checks

The original full environment also includes pinned torch/torchvision CUDA wheels:

```bash
python -m pip install -r environment/requirements-lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130
python -m pytest -q -rs tests
python colab/cpu_smoke_test.py
```

The smoke test exercises profiling and resume logic on CPU; it does not provide
GPU timing evidence. Re-exporting the workload graphs also requires torchvision:

```bash
python scripts/export_graphs.py
(cd configs/workloads && sha256sum -c SHA256SUMS)
```

The exporter writes the manifest files. Run it in a separate checkout when
checking them against the stored version, and inspect `git diff -- configs/workloads`
as well as the hash check. The pinned manifests already suffice for simulation.

## 6. Full experiment queues

Use a dedicated experiment checkout. Queue commands use the study's `final*`
tags and can overwrite outputs at those paths. For exploratory runs, specify a
different `--out` directory as in the quick check above.

```bash
python scripts/final_queues.py
mkdir -p results/logs
export LHAS_PACK_CACHE_DIR="$PWD/results/cache/packings"
scripts/run_list.sh results/queues/chain_main.txt chain_main
scripts/run_list.sh results/queues/graph_main.txt graph_main
scripts/run_list.sh results/queues/gpipe.txt gpipe
scripts/run_list.sh results/queues/ablations.txt ablations
scripts/run_list.sh results/queues/family.txt family
scripts/run_list.sh results/queues/batch.txt batch
scripts/run_list.sh results/queues/sens_chain.txt sens_chain
scripts/run_list.sh results/queues/sens_graph.txt sens_graph
scripts/run_list.sh results/queues/profiled.txt profiled
scripts/run_list.sh results/queues/supp.txt supp
scripts/run_list.sh results/queues/cold.txt cold
```

The supplementary queue includes AlexNet at 1,024 nodes only. The cold-cache queue
clears its dedicated `results/cache/cold` directory before each run. The other
queues may reuse transport packings keyed by the transport inputs and route set.
Cache state affects construction runtime and is recorded in output provenance.

`run_list.sh` logs each command and its exit status in `results/logs/`. It continues
after an unsuccessful command, so inspect every `EXIT` record; the script's final
exit code is not a summary of all runs. An exit of 137 can indicate a resource
termination. Do not fill in missing or unresolved results with inferred values.

An optional construction-cost pilot is available separately:

```bash
python scripts/pilot_allpairs.py 1024:256 1024:512 1024:1024
```

## 7. Measured computation and future profiling

The author's unmodified Tesla T4 ZIP is in
[`results/colab/Tesla_T4_2026-09-28/`](../results/colab/Tesla_T4_2026-09-28/).
To check ingestion without replacing the stored profile:

```bash
LHAS_PROFILE_DIR=$(mktemp -d)
python scripts/ingest_colab.py \
  results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip \
  --out "$LHAS_PROFILE_DIR"
cmp "$LHAS_PROFILE_DIR/profiled_Tesla_T4_B1024.json" \
  configs/profiles/profiled_Tesla_T4_B1024.json
```

Successful `cmp` produces no output. The stored data come from notebook **v2**.
The current **v3** notebook is intended for future profiling and has not been run
on a GPU. Open `colab/LHAS_compute_profiling.ipynb` in Google Colab, select a GPU
runtime, and run all cells. Preserve the downloaded ZIP unchanged, including its
environment and manifest records. A fresh GPU runtime is needed to collect new
timings; downloading or analyzing the existing ZIP is not a new measurement.

Ingestion checks batch size, precision, manifest hashes, local shapes, and timing
records. It does not independently authenticate the physical device or resume
history. Configurations without measured inputs are excluded from every method
in a profiled run. The measured T4 inputs must not be relabeled or rescaled as
P100 measurements. See the [grouped validation protocol](t4_validation_protocol.md)
and [known limitations](STATUS.md).

## Reporting a discrepancy

Include the repository commit, environment, full command, reference JSON path,
and relevant logs in a [reproduction issue](https://github.com/TravisDai/LHAS/issues/new/choose).
Wall-clock runtime differences alone do not imply different modeled costs.
The public code digest and private development commits are explained in
[PROVENANCE.md](../PROVENANCE.md).
