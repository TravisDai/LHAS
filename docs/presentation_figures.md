# Reproducing the compact evaluation figures

These plots present existing results; the script runs no planner, simulation,
or GPU profiling. The frozen snapshot was extracted from the e8f8d15 result
set. On October 5, 2026, all 146 recorded SHA-256 hashes matched the raw
files in public checkpoint `4b608a12a7201837bbc833501bc7e41b51405d67`.

Files:

- `scripts/make_presentation_figures.py` (the plotter)
- `results/processed/presentation_plot_data.json` (the frozen plot-input snapshot)
- `figures/presentation/` (the six figures as PDF, PNG and SVG)
- `docs/presentation_figures.md` (this file)
- `tests/test_presentation_figures.py` (regenerates the snapshot and checks the hashes)

The plotter is the original presentation script with two changes:

- its default output directory is `figures/presentation` (checkpoint 2);
- Figure 6 labels each logarithmic y axis at 1-2-5 steps, so that every
  panel shows at least two values; the original labeled only one tick in
  panels (b)-(d) (checkpoint 3, for legibility in print).

The numerical inputs and the other plotting functions are unchanged.

## Commands

Run from the repository root, using its pinned Python environment. The
plotter needs NumPy and Matplotlib; it does not import torch or the planner.

Re-extract the inputs from the released raw results and draw all six figures:

```bash
python scripts/make_presentation_figures.py --output figures/presentation
```

The command also writes `figures/presentation/presentation_plot_data.json`.
Compare it with the supplied snapshot:

```bash
python - <<'PY'
import json
from pathlib import Path
expected = json.loads(Path('results/processed/presentation_plot_data.json').read_text())
actual = json.loads(Path('figures/presentation/presentation_plot_data.json').read_text())
assert actual == expected, 'The plot input snapshot differs from the frozen results'
print('All plot values, statuses, configurations, and source hashes match.')
PY
```

Alternatively, redraw directly from the supplied snapshot:

```bash
python scripts/make_presentation_figures.py \
  --data results/processed/presentation_plot_data.json \
  --output figures/presentation
```

This second mode reads the snapshot only; it does not independently check
the referenced raw files. To check those files as well:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
d = json.loads(Path('results/processed/presentation_plot_data.json').read_text())
for name, expected in d['source_sha256'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, name
print(f"Verified {len(d['source_sha256'])} raw-result hashes.")
PY
```

## Figure mapping

Figure numbers are those of the paper. Each plot is emitted as PDF, PNG, and
SVG with the listed basename.

| Figure | Output basename | Content |
|---|---|---|
| 6 | `fig_main_comparison_compact` | Predicted iteration times and adapted baselines |
| 7 | `fig_ablations_families` | Planner ablations and collective families |
| 8 | `fig_sensitivity_compact` | Six parameter sensitivities |
| 9 | `fig_batch_comparison` | Best-common-DP/LHAS ratios by global batch |
| 10 | `fig_layer_configurations` | AlexNet and VGG16 layer configurations |
| 12 | `fig_profiled_comparison` | Measured-computation sensitivity versus analytical reference |

Figure 11 is `figures/fig_t4_compute_validation.pdf`, drawn by the
grouped-signature analysis (`python scripts/analyse_profile.py`, input
`results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip`) from its
stored report `results/processed/t4_validation.json`. It is not generated
by this six-figure script. Figures 1-5 of the paper are diagrams, not plots
of results.

## Paper figure and table map

Run the commands from the repository root. The table commands reproduce the
numerical contents in Markdown/CSV; the private manuscript supplies the LaTeX
layout. Tables 1 and 3-6 contain definitions and notation, rather than experimental
results. The worked example (Table 2) is checked by
`python -m pytest -q tests/test_worked_example.py`.

| Paper item | Evidence and raw-result paths | Reproduction command | Generated output |
|---|---|---|---|
| Fig. 6; Table 7 | `results/raw/final/`, `final_gpipe/`, `final_supp/`, `final_supp_gpipe/` | `python scripts/make_presentation_figures.py`; `python scripts/make_tables.py` | `figures/presentation/fig_main_comparison_compact.*`; main comparison in `results/processed/tables.md` |
| Fig. 7; Tables 8-9 | `results/raw/final_ablations/`, `final_family/` | Same two commands | `fig_ablations_families.*`; ablation and family tables |
| Fig. 8; Table 10 | `results/raw/final_sens/` | Same two commands | `fig_sensitivity_compact.*`; sensitivity table |
| Fig. 9; Table 11 | `results/raw/final/`, `final_batch/B256/` | Same two commands | `fig_batch_comparison.*`; batch table |
| Fig. 10; Table 12 | `results/raw/final/`, `final_supp/` | Same two commands | `fig_layer_configurations.*`; configuration table, including supplementary AlexNet |
| Table 13 | `results/raw/final/`, `final_supp/`; operation coverage and recorded runtimes | `python scripts/make_figures.py`; `python scripts/make_tables.py` | Coverage table in `tables.md`; `construction_s` and `plan_s` in `results/processed/main_summary.csv` |
| Fig. 11; Table 14 | Original T4 ZIP in `results/colab/Tesla_T4_2026-09-28/` | `python scripts/analyse_profile.py` | `figures/fig_t4_compute_validation.*`; `results/processed/t4_validation.{json,md}` |
| Fig. 12; Table 15 | `results/raw/final_profiled_T4/`, `final/` | `python scripts/make_presentation_figures.py`; `python scripts/make_tables.py` | `fig_profiled_comparison.*`; profiled comparison table |

Presentation plot basenames in this table are under `figures/presentation/`;
the experiment tags following the first raw path are under `results/raw/`.
The scripts do not rerun the experiments. Reporting scripts rewrite generated
files, so use a separate checkout when comparing regenerated outputs with the
stored version. Use `--output <directory>` for the presentation plotter when you
want its outputs elsewhere. PDF timestamps may differ; compare data and rendered
figures rather than expecting every PDF byte to match.

## Interpretation and verification

- Times remain model predictions. The measured-input comparison combines
  single-GPU local kernel timings with modeled distributed communication.
- Missing/infeasible entries are not replaced by invented numerical values.
  Preserve the manuscript's status qualifications and exact result tables.
- Curves connecting discrete parameter settings are presentation aids,
  not additional evaluated configurations.
- No bootstrap, smoothing, fitting, or retuning is performed by this script.
- Python 3.11.16, NumPy 2.4.4, and Matplotlib 3.10.9 reproduced all six
  original presentation PNGs pixel-for-pixel (before the Figure 6 tick
  change). The fresh snapshot matched the supplied snapshot in every field.
  PDF timestamps can differ even when the rendered figure is identical.

The snapshot's `baseline_commit` identifies its private development origin;
it does not claim that this development commit is present in the public
repository. The per-file SHA-256 map establishes the raw-result identity.

## Check in this repository (2026-10-06, checkpoint 3)

Python 3.11.15, NumPy 2.4.4, Matplotlib 3.10.9: the snapshot re-extracted
from `results/raw` equals the stored snapshot in every field, all 146 raw-file
hashes match, both modes of the script draw identical figures, and the six
PDFs, rendered at 150 dpi, are pixel-identical to Figures 6-10 and 12 of the
revised paper. The Figure 6 tick change alters the y-axis labels only: the
plotted data and the y-axis ranges are unchanged (every data point keeps its
vertical position), and the panels are slightly narrower to make room for
the labels.
`tests/test_presentation_figures.py` repeats the snapshot and hash checks.

The artifact, including these additions, is released under the MIT License
(`LICENSE`).
