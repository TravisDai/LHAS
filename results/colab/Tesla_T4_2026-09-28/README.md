# Colab profile, Tesla T4, 2026-09-28 (run by the author)

`lhas_profile_Tesla_T4.zip`: the unmodified output of `colab/LHAS_compute_profiling.ipynb`
(notebook version 2) on a Google Colab Tesla T4 (driver 580.82.07, CUDA 12.8, cuDNN 9.19,
torch 2.11.0, torchvision 0.26.0; float32, TF32 off). The notebook's manifest check reported a
match for all four workloads. A backup ZIP taken after the profiling cell
(`lhas_profile_Tesla_T4_after_profiling.zip`, sha256 fd196d35…09d51) contained byte-identical
copies of the six files it held and is not stored separately.

Ingested with `python scripts/ingest_colab.py results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip`
→ `configs/profiles/profiled_Tesla_T4_B1024.json`. Analysis: `scripts/analyse_profile.py`.
These are local-computation measurements on one GPU; they do not measure the optical model.
