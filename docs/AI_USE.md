# Use of generative AI tools

This note accompanies the paper's acknowledgment of generative AI use (ACM Policy on
Authorship) and gives the details of that use for this artifact and the paper.

## Claude (Anthropic)

Used, under the authors' direction and from the authors' specification of the model and the
experiments, to:

* review the model specification for consistency;
* develop the analytical-model implementation (`src/lhas/`), the separately implemented
  schedule verifier (`src/lhas_verify/`), the profiling notebook and its generator (`colab/`),
  the experiment runners (`experiments/`), the analysis and table scripts (`scripts/`), and
  their tests (`tests/`);
* prepare the paper's evaluation-workflow diagram (Figure 5), and redraw Figure 2 for print
  from the authors' corrected drawing;
* produce the processed results, the tables and the T4 validation figure (Figure 11) from the
  stored outputs (`results/processed/`, `figures/`);
* adjust Figures 6 and 11 for print (labels of the logarithmic axes in Figure 6, printed size
  of Figure 11); the plotted data did not change;
* draft and edit text of the paper and of this documentation (`README.md`, `docs/`,
  `PROVENANCE.md`).

## ChatGPT (OpenAI)

Used to:

* review the implementation and the manuscript in addition to the authors' own review. These
  included three code audits; their findings and fixes are listed in `docs/change_log.md` and
  `docs/REPORT.md`, and the audit reports themselves are not included;
* develop `scripts/make_presentation_figures.py`, which draws the paper's Figures 6-10 and 12
  from the stored outputs, and its documentation (`docs/presentation_figures.md`);
* edit the language of the paper.

## Human contributions, checks and responsibility

The authors specified the model, the experiments and their parameters, and approved the
modeling decisions (`docs/decision_table.md`). An author ran the profiling notebook on a
Google Colab Tesla T4 to obtain the measurements (`results/colab/`). The authors reviewed and
revised the material produced with these tools.

The code was checked with the test suite, the separately implemented schedule verifier,
exhaustive comparisons on small instances, and selected reruns from a clean revision
(`results/processed/rerun_comparison.md`). Every number reported in the paper was compared with
the stored outputs. None of these checks validates the optical model against physical hardware.
The authors take full responsibility for the model assumptions, the implementation, the analysis
and its interpretation, and the paper.
