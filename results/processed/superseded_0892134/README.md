# Superseded T4 validation outputs (commit 0892134)

Archived unchanged on 2026-09-29. These files used the notebook's split by record ID,
under which 71 timed kernel signatures appeared in both calibration and validation
(finding T1 of `LHAS_0892134_Review.md`), and read several metrics from the notebook's
`device_model_validation.json`. They are kept for traceability and must not be cited.

The current analysis is `scripts/analyse_profile.py` with the grouped protocol of
`docs/t4_validation_protocol.md`; outputs `results/processed/t4_validation.{json,md}` and
`figures/fig_t4_compute_validation.{pdf,png}`. The raw ZIP is unchanged
(sha256 e1648ca8e6dbb6a1…, `results/colab/Tesla_T4_2026-09-28/`).
