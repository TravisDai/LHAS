# Provenance of this public checkpoint

This repository is a public checkpoint of the verified LHAS implementation and evaluation
results. It was prepared on 2026-10-04 from a private development repository at commit
`80f9b070f87c328aad6e733c1b12994d2efa7511` (2026-09-30). The development history also holds
the unpublished manuscript, so it is not published here; the public history starts from a new
root commit. Hashes such as `be19689` below and in the raw outputs refer to development commits
that are therefore not reachable in this repository. The source digest is the public link
between them.

## What this tree contains relative to development commit 80f9b07

* **Identical** (byte for byte): `src/`, `experiments/`, `tests/`, `configs/`, `colab/`,
  `environment/`, `figures/`, `results/raw/`, `results/colab/`, `results/queues/`,
  `results/pilot/`, `results/archive/` (except one withheld file, below), the remaining
  `results/logs/` and `results/processed/` files, `docs/decision_table.md`,
  `docs/memory_ledger.md`, `docs/verification.md`, and every script except the two below.
* **Edited for the public release**: `README.md`; `docs/REPORT.md`, `docs/change_log.md`,
  `docs/experiments.md`, `docs/rerun_dependency_analysis.md` and
  `docs/t4_validation_protocol.md` (manuscript-specific passages, references to withheld files
  and audit-response documents removed; technical content unchanged);
  `scripts/make_tables.py` (the writer of the manuscript's LaTeX tables removed; it still
  writes `results/processed/tables.md` and `table_main.tex`, byte-identical to the stored
  files).
* **Fixed while checking this checkpoint**: `scripts/analyse_profile.py` now puts `src/` on the
  import path. Without it, the documented command `python scripts/analyse_profile.py` stopped
  with `ModuleNotFoundError: lhas` unless `PYTHONPATH` included `src` (the tests set the path
  themselves, so they did not detect it). The analysis itself is unchanged: rerun in a copy of
  this tree, it reproduces `results/processed/t4_validation.{json,md}` and the figure exactly.
  The same one-line fix was applied to the development repository (commit f339b1c).
* **Added**: this file, `docs/STATUS.md`, `scripts/check_reproduction.py` (checkpoint 1; see
  checkpoint 2 below for later additions).
* **Withheld until publication**: the manuscript sources, figures and build scripts, the
  documents answering the code audits, two LaTeX build logs, the generated manuscript numbers
  and tables (`results/processed/manuscript_numbers.json`, `manuscript_tables.tex`), and the
  archived implementation report of commit 757bd4c, which discusses the manuscript. No public
  command or test reads any of them.

No experiment queue was rerun to make this checkpoint, and no raw output, profile, measurement
or provenance record was modified.

## Public checkpoint 2 (2026-10-06)

Added on top of public checkpoint 1 (`4b608a1`); nothing in `src/`, `experiments/`, `configs/`,
`colab/`, `results/raw/`, `results/colab/` or the existing processed outputs changed, so the
source digest and all provenance records below still apply.

* `scripts/make_presentation_figures.py`, `results/processed/presentation_plot_data.json` and
  `docs/presentation_figures.md`: supplied by the authors (written with ChatGPT; see
  `docs/AI_USE.md`). The snapshot records the SHA-256 hash of each of the 146 raw files it reads;
  its `baseline_commit` field names development commit e8f8d15, whose raw results are the files
  in this repository.
* `figures/presentation/*`: drawn from `results/raw` by that script in this repository. The
  re-extracted snapshot equals the stored one in every field, all hashes match, and the PDFs
  rendered at 150 dpi are pixel-identical to the paper's Figures 6-10 and 12.
* `tests/test_presentation_figures.py`, `LICENSE` (MIT), `docs/AI_USE.md`, and updates to
  `README.md`, `docs/STATUS.md` and this file.

## Public checkpoint 3 (2026-10-06)

Added on top of public checkpoint 2 (`b6a2b87`) to make two paper figures legible in print.
Nothing in `src/`, `experiments/`, `configs/`, `colab/`, `results/` or the snapshot
`results/processed/presentation_plot_data.json` changed; the source digest below still applies.

* `scripts/make_presentation_figures.py`: the logarithmic y axes of Figure 6
  (`fig_main_comparison_compact`) are labeled at 1-2-5 steps. The plotted data and the y-axis
  ranges are unchanged. The other five figures are drawn exactly as before.
* `scripts/analyse_profile.py`: Figure 11 (`figures/fig_t4_compute_validation.*`) is drawn at
  its printed size, 5.47 x 2.9 in. The analysis is unchanged: rerun in a copy of this tree, the
  script reproduces `results/processed/t4_validation.{json,md}` byte for byte.
* The regenerated figures are pixel-identical (150 dpi) to Figures 6 and 11 of the revised
  paper, and the other plots to its Figures 7-10 and 12.
* Updates to `docs/presentation_figures.md`, `docs/AI_USE.md`, `docs/STATUS.md`, `README.md`
  and this file.

## Source digests

Every run records `provenance.code_commit` and `provenance.code_dirty`; runs since the audit of
commit 0892134 also record `provenance.source_sha256`, a sha256 over the path and content of
every `src/lhas/**/*.py` and `experiments/*.py` file (`experiments/common.py: source_digest`).
Digests for the earlier commits were computed afterwards from the development repository.

| Development commit | Date | Source digest | Outputs produced from it |
|---|---|---|---|
| `be196897b649` | 2026-09-29 | `af07e85b148ccc02bded3327b29ea017dd42e306749489db0f550759b95cd81b` (recorded in the outputs) | `results/raw/final/` (16), `final_supp/` (1), `final_profiled_T4/` (8); all `code_dirty = false` |
| `80f9b070f87c` | 2026-09-30 | `af07e85b148c…` (same; `src/` and `experiments/` unchanged since be19689) | none (documentation, notebook and analysis scripts only) |
| public checkpoints 1-3 | 2026-10-04 to 2026-10-06 | `af07e85b148c…` (same; checked at each checkpoint) | none |
| `f8bc93e4abec` | 2026-09-28 | `bad9b594b22887a43dd04a4c93f336073b319484f1b6c0046f429a21bd9db6e8` | `final_batch/` (4), `final_family/` (12), `final_sens/` (46 of 92), `final_cold/` (4 of 5) |
| `25e5e52dbf10` | 2026-09-28 | `b25dfa76d701815a3b076f333bfdc8136809c4888b78880ab45ec285e5b081fb` | `final_ablations/` (4), `final_gpipe/` (8), `final_sens/` (46 of 92) |
| `9a7c1d19a4eb` | 2026-09-28 | `b25dfa76d701…` (same source as 25e5e52) | `final_cold/` (1 of 5), `final_supp_gpipe/` (1) |

The outputs of f8bc93e, 25e5e52 and 9a7c1d1 were not rerun after the audit of commit 0892134.
`docs/rerun_dependency_analysis.md` lists, for each later fix, which outputs it can change, and
why these outputs remain valid. Their source differs from be19689 in 12 to 13 files of `src/`
and `experiments/`, and these differences are described in `docs/change_log.md`.

Superseded outputs are kept unchanged for traceability and are not results of the final
specification: `results/raw/archive_0892134/` (primary, supplementary and profiled runs before
the rerun; from development commits 25e5e52, 3b5f5ba, 828f8b5, 9a7c1d1, f5b600f and f8bc93e;
three of these 25 outputs record `code_dirty = true`) and `results/archive/provisional_757bd4c/`
(provisional settings of commit 757bd4c).

`scripts/compare_reruns.py` compares the 25 reruns at be19689 with their archived
predecessors: 25 of 25 pairs, 51 recorded Metropolis incumbent-cost traces, 0 differences
(`results/processed/rerun_comparison.md`).

## Measured T4 computation

`results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip` (sha256
`e1648ca8e6dbb6a199a47e329a13a1f0d04b97a905b66f69cec6bf54f2b161d3`) is the unmodified output of
the profiling notebook, **version 2**, run by the author on a Google Colab Tesla T4 on
2026-09-28. It is ingested as `configs/profiles/profiled_Tesla_T4_B1024.json`; a fresh
ingestion reproduces that file exactly (test `tests/test_colab_safeguards.py`). The notebook now
in `colab/` is **version 3**. It has been checked on CPU only (`colab/cpu_smoke_test.py`) and has
not been run on a GPU, so no stored measurement comes from it.

## Audits

Three independent code audits were made of development commits 0892134, ac66b9e and
e8f8d15. Their findings, fixes and regression tests are summarized in `docs/change_log.md` and
`docs/REPORT.md`; the audit reports and responses are not included.
