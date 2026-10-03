#!/usr/bin/env python
"""Compare rerun outputs with the archived outputs of commit 0892134
(docs/rerun_dependency_analysis.md). Plan costs, configurations and the recorded
Metropolis incumbent-cost traces (iteration, best cost so far) are expected to be
identical; statuses, evaluation counts and elapsed times may change. Complete search-state
trajectories are not recorded and are not compared. Missing or duplicate seeds and
missing required files are failures (nonzero exit).
Writes results/processed/rerun_comparison.md."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OLD = RAW / "archive_0892134"
PAIRS = [("final", "final"), ("final_supp", "final_supp"), ("final_profiled_T4", "final_profiled_T4")]
_W = ("alexnet", "vgg16", "googlenet", "resnet50")
REQUIRED = {"final": [f"{w}_N{n}.json" for w in _W for n in (64, 128, 256, 512)],
            "final_supp": ["alexnet_N1024.json"],
            "final_profiled_T4": [f"{w}_N{n}.json" for w in _W for n in (64, 256)]}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def same(a, b, rel=1e-12):
    a, b = _f(a), _f(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def compare(old: dict, new: dict) -> tuple[list, list]:
    diffs, notes = [], []
    lo, ln = old["lhas"], new["lhas"]
    if not same(lo.get("cost"), ln.get("cost")) or lo.get("configs") != ln.get("configs"):
        diffs.append(f"LHAS {lo.get('cost')} {lo.get('configs')} -> {ln.get('cost')} {ln.get('configs')}")
    if lo.get("status") != ln.get("status"):
        diffs.append(f"LHAS status {lo.get('status')} -> {ln.get('status')}")
    bo, bn = old.get("baselines") or {}, new.get("baselines") or {}
    for k in sorted(set(bo) | set(bn)):
        eo, en = bo.get(k) or {}, bn.get(k) or {}
        if k == "flexflow_mcmc":
            old_runs, new_runs = eo.get("runs", []), en.get("runs", [])
            so, sn = [r.get("seed") for r in old_runs], [r.get("seed") for r in new_runs]
            if len(set(so)) != len(so) or len(set(sn)) != len(sn) or None in so + sn:
                diffs.append(f"Metropolis seed identifiers missing or duplicated: {so} -> {sn}")
            if sorted(so, key=str) != sorted(sn, key=str):
                diffs.append(f"Metropolis seed sets differ: {so} -> {sn}")
            by_seed = {r.get("seed"): r for r in new_runs}
            for ro in old_runs:
                i = ro.get("seed")
                rn = by_seed.get(i)
                if rn is None:
                    continue
                to, tn = ro.get("trace", []), rn.get("trace", [])
                # recorded trace entries: (iteration, evaluations, seconds, best cost); evaluation
                # counts and elapsed times may change (F4 accounting, wall clock), the rest may not
                if len(to) != len(tn) or any(x[0] != y[0] or not same(x[3], y[3]) for x, y in zip(to, tn)):
                    diffs.append(f"Metropolis seed {i}: recorded incumbent-cost trace changed")
                if not same(ro["best_cost"], rn["best_cost"]) or ro.get("best") != rn.get("best") \
                        or ro.get("accepted") != rn.get("accepted"):
                    diffs.append(f"Metropolis seed {i}: {ro['best_cost']} acc {ro.get('accepted')} -> "
                                 f"{rn['best_cost']} acc {rn.get('accepted')}")
                co, cn = ro.get("counts", {}), rn.get("counts", {})
                notes.append(f"Metropolis seed {i}: evaluations {co.get('evaluations')} -> {cn.get('evaluations')} "
                             f"(initialization {rn.get('counts_initialization', {}).get('evaluations')}); "
                             f"infeasible {co.get('infeasible', 0)} -> {cn.get('infeasible', 0)}, not established "
                             f"{co.get('not_established', 0)} -> {cn.get('not_established', 0)}")
            continue
        if k == "ablations":
            continue
        if not same(eo.get("cost"), en.get("cost")):
            diffs.append(f"{k} cost {eo.get('cost')} -> {en.get('cost')}")
        if eo.get("status") != en.get("status"):
            notes.append(f"{k} status {eo.get('status')} -> {en.get('status')}")
        po, pn = eo.get("per_count") or {}, en.get("per_count") or {}
        for c in sorted(set(po) | set(pn), key=lambda x: int(x)):
            if c not in po:          # T6: counts without measured inputs are now enumerated and listed
                notes.append(f"{k} p={c}: added as {(pn.get(c) or {}).get('status')}")
                continue
            so, sn = (po.get(c) or {}).get("status"), (pn.get(c) or {}).get("status")
            if so != sn:
                notes.append(f"{k} p={c}: {so} -> {sn}")
            if not same((po.get(c) or {}).get("cost"), (pn.get(c) or {}).get("cost")):
                diffs.append(f"{k} p={c} cost {(po.get(c) or {}).get('cost')} -> {(pn.get(c) or {}).get('cost')}")
    vo = ((old.get("verification") or {}).get("summary") or {}).get("by_status")
    vn = ((new.get("verification") or {}).get("summary") or {}).get("by_status")
    if vo != vn:
        notes.append(f"verification {vo} -> {vn}")
    return diffs, notes


def main():
    lines = ["# Rerun comparison with the archived outputs of commit 0892134\n",
             "Plan costs and configurations, baseline costs and the recorded Metropolis incumbent-cost traces "
             "(iteration and best cost so far) are expected to be identical (docs/rerun_dependency_analysis.md); "
             "complete search-state trajectories are not recorded. **Differences** list any unexpected change; "
             "*changes* list status and accounting fields that the fixes are expected to change.\n"]
    total_diffs, missing, pairs, traces = 0, [], 0, 0
    for old_tag, new_tag in PAIRS:
        for name in REQUIRED[old_tag]:
            fo, fn = OLD / old_tag / name, RAW / new_tag / name
            absent = [str(f.relative_to(RAW)) for f in (fo, fn) if not f.exists()]
            if absent:
                missing += absent
                continue
            old, new = json.loads(fo.read_text()), json.loads(fn.read_text())
            d, n = compare(old, new)
            total_diffs += len(d)
            pairs += 1
            traces += len(((new.get("baselines") or {}).get("flexflow_mcmc") or {}).get("runs", []))
            pv = new.get("provenance", {})
            lines.append(f"## {new_tag}/{name}\n")
            lines.append(f"New provenance: commit {pv.get('code_commit', '')[:7]}, dirty {pv.get('code_dirty')}, "
                         f"source sha256 {str(pv.get('source_sha256'))[:12]}.\n")
            lines += [f"* **Difference**: {x}" for x in d] or \
                ["* No differences in costs, plans or recorded incumbent-cost traces."]
            lines += [f"* *change*: {x}" for x in n]
            lines.append("")
    lines.insert(2, f"Summary: {pairs} of {sum(len(v) for v in REQUIRED.values())} required pairs compared; "
                    f"{traces} Metropolis incumbent-cost traces; {total_diffs} unexpected differences; missing required files: "
                    f"{', '.join(missing) if missing else 'none'}.\n")
    out = ROOT / "results" / "processed" / "rerun_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(lines[2])
    return total_diffs + len(missing)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
