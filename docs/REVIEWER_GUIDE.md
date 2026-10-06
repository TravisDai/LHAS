# Reviewing the LHAS artifact

This guide is a short route through the evidence accompanying the TOMPECS
submission. It does not imply ACM artifact certification or paper acceptance.
The manuscript and review correspondence are maintained separately.

## 1. Read the results without installing anything

- [Paper figure and table map](presentation_figures.md#paper-figure-and-table-map):
  find a paper item, its raw inputs, reproduction command, and generated output.
- [Exact result tables](../results/processed/tables.md): inspect costs, selected
  node counts, statuses, and the limits of baseline comparisons.
- [T4 report](../results/processed/t4_validation.md): inspect the grouped held-out
  errors, split sensitivity, and within-layer rankings.
- [Specification](decision_table.md): the approved model assumptions and the
  adapted baseline definitions, including their candidate domains.

## 2. Distinguish the evidence

| Evidence | What it supports | What it does not establish |
|---|---|---|
| Predicted distributed iteration times | Comparisons within the declared computation, optical transport, and memory model | Measured distributed-training performance or optical hardware accuracy |
| Local Tesla T4 kernel timings | Computation-model checks on that GPU and a separately identified planning sensitivity | P100 measurements, optical communication measurements, or validation of GPipe/Metropolis comparisons |
| Schedule verifier and small exhaustive tests | Implementation conformance within the recorded test and per-operation coverage | Universal correctness, physical accuracy, or individual verification of every baseline plan |

The T4 sensitivity covers four workloads at modeled ring sizes 64 and 256,
with measured compute inputs, a different reduction rate, and an availability-
restricted candidate domain. It does not isolate the effect of utilization alone.
The stored T4 data come from notebook v2; notebook v3 has been checked on CPU
but has not been run on a GPU.

## 3. Run a bounded reproduction check

Follow the [CPU-only quick start](../README.md#quick-start-cpu-only). It runs
AlexNet at 64 modeled nodes in a separate output directory and checks the LHAS,
DP, best-common-DP, and OWT results against the stored JSON. Expected outcome:
`RESULT: agrees`. It does not rerun the full queues, GPipe, or Metropolis.

The [reproduction guide](REPRODUCING.md#3-inspect-and-regenerate-reports) also
provides commands for regenerating figures and reports, comparing the 25 stored
reruns, and checking ingestion of the original T4 ZIP. Report test skip reasons,
not just the pass count.

## 4. Follow provenance and qualifications

Record `git rev-parse HEAD`. The public history begins at a clean checkpoint;
development hashes in stored results refer to the private development history.
[PROVENANCE.md](../PROVENANCE.md) supplies the source digests and the mapping.
Do not relabel old outputs with the current documentation commit.

Read [verification.md](verification.md) for actual-payload, reduced-payload,
first-fit, and skipped-operation coverage. Read [STATUS.md](STATUS.md) for work
not performed. An `infeasible` result is different from `not_established`,
`incumbent`, or `unavailable`; the [status definitions](REPRODUCING.md#4-understand-statuses-and-verification)
explain how to interpret each one. Branch-search optimality is conditional on
the represented additive objective and the fastest plan fitting the memory budget.

## 5. Report a discrepancy

Use the [reproduction issue form](https://github.com/TravisDai/LHAS/issues/new/choose)
with the commit, environment, command, raw JSON path, and observed versus expected
output. For a figure discrepancy, include the figure basename and snapshot/source
hash check. Construction runtime can change across machines and cache states
without changing the modeled iteration cost.
