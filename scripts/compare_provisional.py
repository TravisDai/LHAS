#!/usr/bin/env python
"""Old (provisional, commit 757bd4c, archived) vs final predicted times, for the report's
"numerical changes" section. The two sets differ in assumptions (raw-input gradient,
momentum state, runtime reserve, memory-ledger corrections, BatchNorm state, baseline
definitions), so differences are not attributable to one change. Writes
results/processed/provisional_vs_final.md."""
from __future__ import annotations

import csv
import math

from results_lib import ROOT  # noqa: E402

OLD = ROOT / "results" / "archive" / "provisional_757bd4c" / "processed" / "main_summary.csv"
NEW = ROOT / "results" / "processed" / "main_summary.csv"


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def ms(v):
    return "—" if v != v else f"{v * 1e3:.1f}"


def main():
    old = {(r["workload"], int(r["N"])): r for r in csv.DictReader(open(OLD))}
    new = {(r["workload"], int(r["N"])): r for r in csv.DictReader(open(NEW))}
    cols = [("lhas", "lhas"), ("dp_N", "dp_N"), ("dp_best", "dp_best"), ("owt", "owt"),
            ("flexflow_mcmc", "flexflow_mcmc"), ("gpipe_proposed", "gpipe")]
    out = ["# Provisional (757bd4c) vs final predicted time per iteration (ms)\n",
           "Provisional: raw-input gradient retained, no optimizer state, no runtime reserve, earlier memory "
           "ledger and baseline definitions. Final: specification D1–D17. Each cell: provisional → final "
           "(ratio final/provisional).\n",
           "| workload | N | LHAS | DP (p=N) | DP best | OWT | FlexFlow-MCMC | GPipe | LHAS configs changed |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(set(old) | set(new), key=lambda k: (["alexnet", "vgg16", "googlenet", "resnet50"].index(k[0]), k[1])):
        o, n = old.get(key), new.get(key)
        cells = []
        for co, cn in cols:
            a = f(o.get(co)) if o else math.nan
            b = f(n.get(cn)) if n else math.nan
            r = f" ({b / a:.2f})" if a == a and b == b and a > 0 else ""
            cells.append(f"{ms(a)} → {ms(b)}{r}")
        ch = "—" if not (o and n) else ("no" if o.get("lhas_configs") == n.get("lhas_configs") else "yes")
        out.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + f" | {ch} |")
    (ROOT / "results" / "processed" / "provisional_vs_final.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
