#!/usr/bin/env python
"""Compare a freshly computed run with the stored result of the same workload and system size.

Usage: python scripts/check_reproduction.py <new.json> [--ref-tag final]

Checked: LHAS status, selected configurations and cost; status and cost of every baseline
present in both files (and the node count of best-common DP); the manifest and nominal-
specification hashes; and that the Python source digest of the new run equals the one recorded
in the stored result. Metropolis searches are compared only if both files contain them.
Costs must agree to a relative tolerance of 1e-12. Exit status 0 if everything agrees.
Use --require-baselines to require named JSON baseline keys in both files. CI may
use --allow-source-change to check numerical regression across source edits; this
explicitly skips source-digest equality, not model inputs or result comparisons."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("--ref-tag", default="final")
    ap.add_argument("--require-baselines", default="",
                    help="Comma-separated JSON keys required in both runs (e.g. dp_N,dp_best,owt)")
    ap.add_argument("--allow-source-change", action="store_true",
                    help="Skip source-digest equality for numerical regression across code changes")
    a = ap.parse_args()
    new = json.loads(Path(a.new).read_text())
    ref_path = ROOT / "results" / "raw" / a.ref_tag / f"{new['workload']}_N{new['N']}.json"
    ref = json.loads(ref_path.read_text())
    problems: list[str] = []

    def same(label, x, y, tol=False):
        ok = (x == y) if not tol else (x is not None and y is not None and math.isclose(x, y, rel_tol=1e-12, abs_tol=0.0))
        print(f"{'ok  ' if ok else 'DIFF'} {label}: new={x!r} stored={y!r}" if not ok or tol else f"ok   {label}")
        if not ok:
            problems.append(label)

    for k in ("schema", "engine", "B", "family", "settings", "transport", "compute", "input_mode"):
        same(k, new.get(k), ref.get(k))
    for k in ("manifest_sha256", "nominal_spec_sha256", "source_sha256"):
        if k == "source_sha256" and a.allow_source_change:
            print("SKIP provenance.source_sha256 equality (--allow-source-change): "
                  f"new={new['provenance'].get(k)!r} stored={ref['provenance'].get(k)!r}")
            continue
        same(f"provenance.{k}", new["provenance"].get(k), ref["provenance"].get(k))
    same("lhas.status", new["lhas"]["status"], ref["lhas"]["status"])
    same("lhas.configs", new["lhas"]["configs"], ref["lhas"]["configs"])
    same("lhas.cost (s)", new["lhas"]["cost"], ref["lhas"]["cost"], tol=True)
    required = {b.strip() for b in a.require_baselines.split(",") if b.strip()}
    for b in sorted(required):
        same(f"{b}.present_in_new", b in new["baselines"], True)
        same(f"{b}.present_in_stored", b in ref["baselines"], True)
    common = sorted(set(new["baselines"]) & set(ref["baselines"]))
    for b in common:
        nb, rb = new["baselines"][b], ref["baselines"][b]
        same(f"{b}.status", nb.get("status"), rb.get("status"))
        cost_key = "best_cost" if "best_cost" in rb else "cost"
        same(f"{b}.{cost_key} (s)", nb.get(cost_key), rb.get(cost_key), tol=True)
        if "p" in rb:
            same(f"{b}.p", nb.get("p"), rb.get("p"))
    skipped = sorted(set(ref["baselines"]) - set(new["baselines"]))
    print(f"stored result: {ref_path.relative_to(ROOT)} (code_commit {ref['provenance'].get('code_commit', '?')[:7]})")
    if skipped:
        print("baselines in the stored result that were not rerun:", ", ".join(skipped))
    print("RESULT:", "agrees" if not problems else f"{len(problems)} difference(s): {', '.join(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
