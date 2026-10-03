# T4 compute-model validation: grouping, weighting and split protocol

Declared on 2026-09-29, before the corrected statistics were computed, in response to
finding T1 of `LHAS_0892134_Review.md`. This is not a preregistration: the reviewer's
illustrative statistics for a string-sorted variant of the grouped split were known when the
protocol was written, and the protocol and its first results were committed together
(4cff71f). Implemented in `scripts/analyse_profile.py`
(constants `SIGNATURE_FIELDS`, `PROTOCOL`); regression
`tests/test_review_0892134.py::test_t1_grouped_split_has_no_shared_kernel_signatures`.

## Why the earlier split was wrong

The notebook (version 2) keyed a profiled record by the parent layer's full output width
`n_out_full` as well as by the dimensions of the kernels it times. `n_out_full` does not
enter the timed kernels: an MP partition of a 128-wide layer into 2 and a DP partition of a
64-wide layer at the same local batch time the same kernels. The 1,759 measured records
therefore contain 1,542 distinct timed kernel shapes, and the notebook's split by record ID
placed 71 such shapes on both sides (116 of the 880 validation records had a calibration
twin). The timing samples themselves are valid; only the grouping was wrong.

## Protocol

| Item | Definition |
|---|---|
| Unit of analysis | A **timed kernel signature**: `(kind, x, n_out_local, batch_local, kernel, stride, padding, bias)`, with lists as integer tuples and `bias` as a Boolean. Device (Tesla T4), dtype (float32) and TF32 policy (off) are constant in this ZIP and are asserted, not keyed. Dilation and groups are 1 for every profiled layer; they must be added to the signature if a later manifest uses other values. |
| Records | Rows of `layer_timings_summary.csv` with status `measured`, after keeping the last row per `shape_id` (the rule used by `scripts/ingest_colab.py` for resumed runs; this ZIP has no repeated IDs). |
| Record time | Sum of the forward, input-gradient and weight-gradient median times of the record. |
| Signature time | Median of the record times of the records with that signature. |
| Weighting | Each signature counts once, regardless of how many records or layer/configuration uses share it. |
| Split | Signatures sorted by their typed value, shuffled with `random.Random(0)` (the notebook's `SEED`), first `floor(n/2)` for calibration, the remainder for validation. No signature is on both sides (asserted). |
| Estimator | `R = sum(3F) / sum(T)` over the calibration signatures, `F` = FLOPs per product; the notebook's aggregate estimator, unchanged. Prediction `3F/R`. |
| Metrics on validation | Absolute relative error `|pred − meas| / meas`: median, 90th and 95th percentiles, maximum. Signed relative error `(pred − meas) / meas`: median, and fraction overpredicted. Weighted absolute error `sum|pred − meas| / sum(meas)`. Error of the summed time `(sum pred − sum meas) / sum(meas)`. Global Spearman rank correlation of predicted and measured time. All metrics also by kind (CONV, FC). Percentiles use linear interpolation between order statistics. |
| Split variability | The same protocol with seeds 0–99; median and 5th–95th percentiles over seeds. Seed 0 is the reported split; the others only show its sensitivity. |
| Configuration ranking | Split-independent (the rate `R` cancels). Within each (workload, layer), every pair of measured configurations, with the time and local FLOPs the nominal planner uses for each configuration (forward and weight-gradient products, plus the input-gradient product except for layers that consume the raw graph input, whose input gradient the nominal planner omits; corrected after review ac66b9e, A1): pairs whose local FLOPs differ are **concordant** if the measured order agrees with the FLOP order; pairs of the same strategy at consecutive measured counts are reported separately; pairs with equal local FLOPs (DP(p) and MP(p)) cannot be ordered by the model, and their measured ratio is reported; the p = 1 pairs, where DP and MP time the same kernel, are counted and also reported separately (added after review ac66b9e). |
| Scaling efficiency | For DP configurations of a layer, `T(p0) p0 / (T(p) p)` with `p0` the smallest measured DP count of that layer and `T` the planner time as for the ranking. Layers with `p0 > 1` are listed: their reference is not the full batch. |
| Repeatability | Per record and product (forward, input gradient, weight gradient): interquartile range over median of the 30 repetitions after 10 warm-up iterations. It describes repetition within one session on one GPU, not run-to-run or device-to-device variation. |
| Model comparison | Zero-intercept and affine models are compared on the same calibration signatures under the same criterion (least squares of relative residuals), in addition to the primary aggregate estimator. |

## Outputs

`results/processed/t4_validation.json` is the only source of the numbers: the Markdown
report and the figure are generated from that file, never from the
notebook's `device_model_validation.json`. The notebook's record-ID split is recomputed from
the CSV only to document the leakage; it is labeled superseded. The previous outputs are
archived unchanged in `results/processed/superseded_0892134/`.

## Scope

Local computation on one Tesla T4 in float32 without TF32. Nothing here measures the optical
network, the P100 reference, multi-GPU execution or iteration time of a network; the summed
errors are over kernel-signature samples, not over any network's iteration.
