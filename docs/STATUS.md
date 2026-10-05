# Status of the artifact: outstanding work and open items

Public checkpoint 2 of 2026-10-06 (see `PROVENANCE.md`). Items that concern only the unpublished
manuscript are tracked privately and are not listed here.

## Open items from the code audits

* **Resume authentication.** The notebook (v3) refuses to resume under a changed
  configuration, hardware class, driver, CUDA, framework or precision policy, but the
  fingerprint does not authenticate a physical GPU. `scripts/ingest_colab.py` requires a
  nonempty resume-fingerprint marker for v3 ZIPs; it does not independently verify the resume
  history. Stronger authentication was not implemented.
* **Metropolis traces.** Reruns are compared on recorded incumbent-cost traces (iteration and
  best cost so far). Complete search-state trajectories are not recorded, so equality of
  trajectories is not claimed.
* **Equal-FLOP statistic.** Of the 1,066 DP(p)/MP(p) pairs with equal local FLOPs, 130 have
  p = 1 and time the same kernel. The declared statistic over all pairs (median ratio 3.05) and
  the p > 1 statistic (936 pairs, median 3.77) are both reported in
  `results/processed/t4_validation.{json,md}`.

## Not run or not verified

* Notebook v3 on a GPU. All stored T4 measurements come from notebook v2.
* The optional N = 1024 runs of VGG16, GoogLeNet and ResNet-50 (AlexNet N = 1024 is the only
  supplementary large-N run).
* Reruns of the outputs from development commits f8bc93e, 25e5e52 and 9a7c1d1 (sensitivity,
  families, batch, cold cache, ablations, GPipe). They predate `source_sha256` recording;
  `docs/rerun_dependency_analysis.md` explains why later fixes cannot change them.
* Any validation of the optical model against hardware or an independent simulator.
* The GPipe adaptation selects its largest allowed microbatch count (M = 32) in 8 of 9 cases;
  a larger range could lower GPipe times and was not explored.

## Release

* **License:** MIT (`LICENSE`).
* **Archival release:** none yet (no tagged release, no Zenodo deposit).
* **Paper figures:** the plotting script for Figures 6-10 and 12 is included and reproduces them
  from the stored results (`docs/presentation_figures.md`).
* `configs/approved_reference_parameters.json` (frozen, author-approved) still carries its
  original status note about unpinned inputs; `configs/nominal_experiment.json` pins them.
