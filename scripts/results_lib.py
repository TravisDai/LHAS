"""Reading final (schema lhas-run-v2 / lhas-gpipe-v2) result files.

Every planner/baseline entry is reduced to (value in seconds or NaN, status):
  feasible         value is a completed, budget-satisfying result
  incumbent        best completed result of a set with unresolved members (D11)
  resource-limited Metropolis search in which some proposals were not established (D14): chain
                   runner, stopped by the label cap; branch runner, fastest schedule combination
                   failed the budget without proof
  infeasible       proven: no schedule combination satisfies the budget
  not_established  unresolved (label cap, or the fastest combination fails without proof)
  unavailable      a required measured compute input is missing (profiled mode only)
  missing          run absent
Only feasible, incumbent and resource-limited entries carry a value."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
WORKLOADS = ["alexnet", "vgg16", "googlenet", "resnet50"]
CHAIN = ("alexnet", "vgg16")
NS = [64, 128, 256, 512]
NOTE = ("Predictions of the analytical model (hypothetical P100 PCIe compute reference; approved optical "
        "parameters; B = 1024); not measurements.")


def load(tag, w, N):
    f = RAW / tag / f"{w}_N{N}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def value(e):
    """(seconds or nan, status) for a v2 entry."""
    if e is None:
        return math.nan, "missing"
    st = e.get("status", "")
    if "best_cost" in e:                                   # Metropolis search
        c = _f(e["best_cost"])
        if not math.isfinite(c):
            return math.nan, "not_established"
        limited = any((r.get("counts") or {}).get("not_established", 0) > 0 for r in e.get("runs", []))
        return c, "resource-limited" if limited else "feasible"
    c = _f(e.get("cost"))
    if st in ("feasible", "incumbent") and math.isfinite(c):
        return c, st
    if st in ("infeasible", "not_established", "unavailable"):
        return math.nan, st
    return (c, "feasible") if math.isfinite(c) else (math.nan, st or "infeasible")


def gpipe_value(w, N, tag="final_gpipe"):
    d = load(tag, w, N)
    if d is None:
        return math.nan, "missing", None
    r = d["result"]
    v, s = value(r)
    return v, s, r


def fmt_ms(v, s, digits=1):
    if s in ("feasible",):
        return f"{v * 1e3:.{digits}f}"
    if s == "incumbent":
        return f"{v * 1e3:.{digits}f}†"
    if s == "resource-limited":
        return f"{v * 1e3:.{digits}f}‡"
    return {"infeasible": "infeasible", "not_established": "n.e.", "unavailable": "unavail.", "missing": "—"}.get(s, s)


def fmt_tex(v, s, digits=1):
    if s == "feasible":
        return f"{v * 1e3:.{digits}f}"
    if s == "incumbent":
        return f"{v * 1e3:.{digits}f}$^\\dagger$"
    if s == "resource-limited":
        return f"{v * 1e3:.{digits}f}$^\\ddagger$"
    return {"infeasible": "infeas.", "not_established": "n.e.", "unavailable": "unavail.", "missing": "--"}.get(s, s)
