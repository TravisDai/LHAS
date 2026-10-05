# Use of generative AI tools

This note accompanies the paper's acknowledgment of generative AI use (ACM Policy on
Authorship) and describes the use for this artifact.

## Claude (Anthropic)

Used, under the authors' direction and from the authors' specification of the model and the
experiments, to:

* check the analytical model for consistency;
* implement the analytical model (`src/lhas/`), the independent verifier (`src/lhas_verify/`),
  the experiment runners (`experiments/`), the analysis and table scripts (`scripts/`), the
  profiling notebook and its generator (`colab/`), and the tests (`tests/`);
* generate the processed results, tables and the T4 validation figure from the stored raw
  results (`results/processed/`, `figures/`);
* draft and edit the documentation (`README.md`, `docs/`, `PROVENANCE.md`).

## ChatGPT (OpenAI)

Used to:

* review the implementation and the manuscript independently. The audit findings and their
  fixes are listed in `docs/change_log.md` and `docs/REPORT.md`; the audit reports themselves
  are not included;
* write `scripts/make_presentation_figures.py`, which draws the paper's Figures 6-10 and 12
  from the stored results, and its documentation (`docs/presentation_figures.md`);
* edit the language of the paper.

## Human checks and responsibility

The authors specified the model, the experiments and their parameters, and approved the
modeling decisions (`docs/decision_table.md`). Generated code was checked with the test suite,
the independent verifier, exhaustive comparisons on small instances, and reruns from a clean
revision (`results/processed/rerun_comparison.md`). Every reported number was checked against
the stored outputs. The Tesla T4 measurements were made by the authors on Google Colab
(`results/colab/`). The authors take full responsibility for the artifact and the paper.
