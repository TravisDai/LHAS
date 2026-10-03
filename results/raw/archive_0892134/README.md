# Superseded raw outputs (up to commit 0892134)

Archived unchanged on 2026-09-29 before the reruns for the findings of
`LHAS_0892134_Review.md`; see `docs/rerun_dependency_analysis.md`.

* `final/` (16 primary runs) and `final_supp/alexnet_N1024.json`: Metropolis outputs lack the
  initialization accounting (F4); graph Metropolis counters classify some unresolved
  proposals as infeasible (F1); three runs had uncommitted source (`code_dirty`).
* `final_profiled_T4/`: best-common DP labels counts without measured compute inputs as
  infeasible instead of unavailable (T6).

Plan costs are expected to be unchanged by the fixes; the reruns check this exactly.
