#!/usr/bin/env python
"""Print the key comparisons used in the manuscript text (from results/processed/main_summary.csv
and the raw result files). Ratios are baseline time / LHAS time (>1: LHAS faster)."""
from __future__ import annotations

import csv
import math

from results_lib import ROOT, WORKLOADS, CHAIN, load, value  # noqa: E402

rows = list(csv.DictReader(open(ROOT / "results" / "processed" / "main_summary.csv")))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def ratios(key, prim=True):
    out = {}
    for r in rows:
        if prim and r["supplementary"] == "True":
            continue
        a, b = f(r["lhas"]), f(r[key])
        if a == a and b == b:
            out[(r["workload"], int(r["N"]))] = b / a
    return out


for key in ("dp_N", "dp_best", "owt", "flexflow_mcmc", "gpipe"):
    rs = ratios(key)
    if not rs:
        continue
    print(f"\n{key}: max {max(rs.values()):.3f} at {max(rs, key=rs.get)}, min {min(rs.values()):.3f} at {min(rs, key=rs.get)}")
    for w in WORKLOADS:
        v = {N: round(x, 3) for (ww, N), x in rs.items() if ww == w}
        if v:
            print(f"  {w}: {v}")
print("\nLHAS ms and status:")
for r in rows:
    print(f"  {r['workload']} N={r['N']}: {f(r['lhas']) * 1e3:.2f} {r['lhas_status']} | dp_best p={r['dp_best_p']} "
          f"{r['dp_best_status']} | mcmc {r['flexflow_mcmc_status']} ne={r['mcmc_not_established_evaluations']} "
          f"| gpipe {r['gpipe_status']} | mem {f(r['lhas_memory_max_GiB']):.2f} GiB | verify {r['verify_by_status']}")
print("\nconfigs:")
for r in rows:
    if r["workload"] in CHAIN:
        print(f"  {r['workload']} N={r['N']}: {r['lhas_configs']}")
    else:
        cs = r["lhas_configs"].split()
        from collections import Counter
        print(f"  {r['workload']} N={r['N']}: {dict(Counter(cs))}")
