# Documentation guide

Start with the [project README](../README.md) for the study scope and a CPU-only
reproduction check. The pages below provide progressively more detail.

## Reproduce and interpret the study

| Document | Purpose |
|---|---|
| [Reviewer guide](REVIEWER_GUIDE.md) | A short path through the evidence, reproducibility checks, and limits |
| [Paper figure and table map](presentation_figures.md#paper-figure-and-table-map) | Figure numbers, commands, inputs, and generated outputs |
| [Reproduction guide](REPRODUCING.md) | Installation, a small reproduction check, report generation, optional checks, and full experiment queues |
| [Experiments](experiments.md) | Experiment IDs, workloads, system sizes, baseline coverage, and output directories |
| [Result tables](../results/processed/tables.md) | Exact modeled costs, statuses, selected configurations, and verification coverage |
| [T4 validation report](../results/processed/t4_validation.md) | Measurement coverage, held-out errors, ranking results, and sensitivity to the split |
| [Project status](STATUS.md) | Known limitations, work not performed, and release status |

## Inspect the model and evidence

| Document | Purpose |
|---|---|
| [Experimental specification](decision_table.md) | Decisions D1-D17, model contracts, baseline definitions, and implementation locations |
| [Memory ledger](memory_ledger.md) | Reservations, tensor ownership, aliasing, and memory-feasibility rules |
| [Verification](verification.md) | Independent checks, selected-plan coverage, and the limits of the evidence |
| [T4 validation protocol](t4_validation_protocol.md) | Kernel signatures, grouped splits, estimators, and reported statistics |
| [Provenance](../PROVENANCE.md) | Public checkpoint, private development commits, source digests, and measurement origin |

## Development and maintenance records

- [Implementation report](REPORT.md): recorded execution and verification outcomes.
- [Change log](change_log.md): model-to-code mapping and fixes from code audits.
- [Rerun dependency analysis](rerun_dependency_analysis.md): which corrections
  required new runs and why other outputs were retained.
- [Rerun comparison](../results/processed/rerun_comparison.md): comparisons with
  the archived predecessor outputs.
- [AI-use documentation](AI_USE.md): tools, author contributions, and checks.
- [Contributing](../CONTRIBUTING.md) and [citation guidance](../CITATION.md).

References to a "reviewer" or an audit ID in the development records refer to
code audits, not an ACM artifact badge or a publication-acceptance decision.
The unpublished manuscript and its review correspondence are maintained separately.
