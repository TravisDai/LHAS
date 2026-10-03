#!/usr/bin/env python
"""Markdown and LaTeX tables of the final results (results/processed/tables.md,
results/processed/table_main.tex). Values: predicted time per iteration in ms.
Marks: † incumbent (some DP counts unresolved), ‡ resource-limited Metropolis search
(some proposals not established), 'n.e.' not established, 'infeasible' proven,
'—' not run."""
from __future__ import annotations

import csv
import json
import math

from results_lib import ROOT, RAW, WORKLOADS, CHAIN, NS, load, value, gpipe_value, fmt_ms, fmt_tex  # noqa: E402

PROC = ROOT / "results" / "processed"
HEAD = ("Predicted time per training iteration (ms) from the analytical model: hypothetical P100 PCIe compute "
        "reference, approved optical parameters, B = 1024, synchronized BatchNorm, raw-input gradient omitted, SGD "
        "momentum state, 1 GiB runtime reserve inside the 12 GiB budget. † incumbent (best completed result, some "
        "DP counts unresolved); ‡ Metropolis search with unresolved proposals; n.e. = not established; "
        "infeasible = proven by exact search or reservation lower bound; — = not run.\n")


def main_table(rows):
    out = ["| workload | N | LHAS | DP (p=N) | DP best (p) | OWT | FlexFlow-MCMC | GPipe | LHAS vs DP(N) | LHAS vs best baseline | LHAS status |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for r in rows:
        best_bl = min([float(r[k]) for k in ("dp_N", "dp_best", "owt", "flexflow_mcmc", "gpipe")
                       if r[k] not in ("", "nan") and not math.isnan(float(r[k]))], default=math.nan)
        lh = float(r["lhas"]) if r["lhas"] not in ("", "nan") else math.nan
        dpn = float(r["dp_N"]) if r["dp_N"] not in ("", "nan") else math.nan
        sp1 = f"{dpn / lh:.2f}×" if lh == lh and dpn == dpn else "—"
        sp2 = f"{best_bl / lh:.2f}×" if lh == lh and best_bl == best_bl else "—"
        cells = {k: fmt_ms(float(r[k]) if r[k] not in ("",) else math.nan, r[k + "_status"])
                 for k in ("lhas", "dp_N", "dp_best", "owt", "flexflow_mcmc", "gpipe")}
        if r["gpipe_status"] == "not applicable":
            cells["gpipe"] = "n/a"
        p = r["dp_best_p"] or "—"
        sup = " (suppl.)" if r["supplementary"] == "True" else ""
        out.append(f"| {r['workload']} | {r['N']}{sup} | {cells['lhas']} | {cells['dp_N']} | {cells['dp_best']} ({p}) | "
                   f"{cells['owt']} | {cells['flexflow_mcmc']} | {cells['gpipe']} | {sp1} | {sp2} | {r['lhas_status']} |")
        t = {k: fmt_tex(float(r[k]) if r[k] not in ("",) else math.nan, r[k + "_status"])
             for k in ("lhas", "dp_N", "dp_best", "owt", "flexflow_mcmc", "gpipe")}
        if r["gpipe_status"] == "not applicable":
            t["gpipe"] = "n/a"
        sp_tex = sp1.replace("×", r"$\times$")
        star = "$^{*}$" if sup else ""
        tex.append(f"{r['workload']} & {r['N']}{star} & {t['lhas']} & {t['dp_N']} & {t['dp_best']} ({p}) & "
                   f"{t['owt']} & {t['flexflow_mcmc']} & {t['gpipe']} & {sp_tex} " + r"\\")
    return out, tex


def ablation_table():
    out = ["| workload | N | LHAS | strategy only (fixed counts) | DP only, per-layer counts | MP only, per-layer counts | greedy layer-wise |",
           "|---|---|---|---|---|---|---|"]
    for w in CHAIN:
        for N in (64, 256):
            d = load("final_ablations", w, N)
            if d is None:
                out.append(f"| {w} | {N} | — | — | — | — | — |")
                continue
            a = d["baselines"].get("ablations", {})
            c = [fmt_ms(*value(a.get(k))) for k in ("strategy_only", "dp_only_counts", "mp_only_counts", "greedy_layerwise")]
            out.append(f"| {w} | {N} | {fmt_ms(*value(d['lhas']))} | " + " | ".join(c) + " |")
    return out


def gpipe_table():
    out = ["| workload | N | GPipe | S | R | active nodes | M | remat | degenerate DP | status | evaluated / candidates |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in CHAIN:
        for N in NS + ([1024] if w == "alexnet" else []):
            v, s, r = gpipe_value(w, N, "final_gpipe" if N != 1024 else "final_supp_gpipe")
            if r is None:
                out.append(f"| {w} | {N} | — |  |  |  |  |  |  | missing | |")
                continue
            d = load("final_gpipe" if N != 1024 else "final_supp_gpipe", w, N)
            st = d.get("stats", {})
            out.append(f"| {w} | {N} | {fmt_ms(v, s)} | {r.get('S', '')} | {r.get('R', '')} | {r.get('active_nodes', '')} | "
                       f"{r.get('M', '')} | {r.get('rematerialization', '')} | {r.get('degenerate_dp', '')} | {s} | "
                       f"{st.get('evaluated')} / {st.get('candidates')} |")
    return out


def batch_table():
    out = ["| workload | B | N | LHAS | DP (p=N) | DP best (p) | OWT |", "|---|---|---|---|---|---|---|"]
    for w in WORKLOADS:
        for tag, B in (("final", 1024), ("final_batch/B256", 256)):
            d = load(tag, w, 64)
            if d is None:
                out.append(f"| {w} | {B} | 64 | — | — | — | — |")
                continue
            bl = d["baselines"]
            db = bl.get("dp_best") or {}
            out.append(f"| {w} | {B} | 64 | {fmt_ms(*value(d['lhas']))} | {fmt_ms(*value(bl.get('dp_N')))} | "
                       f"{fmt_ms(*value(db))} ({db.get('p', '—')}) | {fmt_ms(*value(bl.get('owt')))} |")
    return out


def family_table():
    out = ["| workload | all families | one-stage | deepest |", "|---|---|---|---|"]
    for w in WORKLOADS:
        c = []
        for f in ("all", "one-stage", "deepest"):
            d = load(f"final_family/{f}", w, 64)
            c.append(fmt_ms(*value(d["lhas"])) if d else "—")
        out.append(f"| {w} | " + " | ".join(c) + " |")
    return out


def sens_table():
    base = RAW / "final_sens"
    if not base.exists():
        return []
    settings = sorted(p.name for p in base.iterdir())
    out = ["| setting | " + " | ".join(WORKLOADS) + " |", "|---|" + "---|" * len(WORKLOADS)]
    for s in settings:
        cells = []
        for w in WORKLOADS:
            d = load(f"final_sens/{s}", w, 64)
            if d is None:
                cells.append("—")
                continue
            bl = d.get("baselines", {})
            cells.append(f"{fmt_ms(*value(d['lhas']))} / {fmt_ms(*value(bl.get('dp_N')))} / {fmt_ms(*value(bl.get('owt')))}")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    return out


def verification_table(rows):
    out = ["| workload | N | selected operations | network operations | checked (actual) | checked (reduced payload) | skipped | packing re-derived |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bs = dict(x.split(":") for x in r["verify_by_status"].split()) if r["verify_by_status"] else {}
        out.append(f"| {r['workload']} | {r['N']} | {r['verify_expected_ops']} | {r['verify_network_ops']} | "
                   f"{bs.get('checked-actual', 0)} | {bs.get('checked-reduced', 0)} | {bs.get('skipped', 0)} | "
                   f"{r.get('verify_packing_checked', '—')} |")
    return out


def mcmc_table(rows):
    """Metropolis accounting (F4): per-seed distinct evaluations summed over the seeds,
    initialization included, globally distinct plans, and rejected evaluations."""
    out = ["| workload | N | evaluations (sum over seeds) | of which initialization | globally distinct plans | "
           "not established (rejected) | seconds (sum over seeds) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['workload']} | {r['N']} | {r.get('mcmc_evaluations', '')} | "
                   f"{r.get('mcmc_evaluations_initialization', '')} | {r.get('mcmc_evaluations_globally_distinct', '')} | "
                   f"{r.get('mcmc_not_established_evaluations', '')} | "
                   f"{float(r['mcmc_seconds_sum']):.0f} |" if r.get("mcmc_seconds_sum") else
                   f"| {r['workload']} | {r['N']} | — | — | — | — | — |")
    return out


def profiled_table():
    names = {"alexnet": "AlexNet", "vgg16": "VGG16", "googlenet": "GoogLeNet", "resnet50": "ResNet50"}
    out = ["| workload | N | LHAS (ms) | DP (p=N) | DP best | OWT | DP best certification | unavailable DP counts |",
           "|---|---|---|---|---|---|---|---|"]
    for w in WORKLOADS:
        for N in (64, 256):
            p, a = load("final_profiled_T4", w, N), load("final", w, N)
            if p is None or a is None:
                continue
            lp, la = value(p["lhas"])[0], value(a["lhas"])[0]
            cells = [f"{value(p['baselines'][k])[0] / lp:.2f} ({value(a['baselines'][k])[0] / la:.2f})"
                     for k in ("dp_N", "dp_best", "owt")]
            db = p["baselines"]["dp_best"]
            out.append(f"| {names[w]} | {N} | {lp * 1e3:.1f} ({la * 1e3:.1f}) | " + " | ".join(cells)
                       + f" | {db.get('certification', '—')} | {' '.join(map(str, db.get('unavailable_counts', [])))} |")
    return out


def main():
    f = PROC / "main_summary.csv"
    rows = list(csv.DictReader(open(f))) if f.exists() else []
    md, tex = main_table(rows)
    out = [HEAD, "### Main comparison (N = 64-512; AlexNet N = 1024 supplementary)\n", *md,
           "\n### GPipe (adapted, contract D13)\n", *gpipe_table(),
           "\n### Planner ablations\n", *ablation_table(),
           "\n### Collective-family comparison at N = 64 (LHAS re-planned with each candidate family)\n", *family_table(),
           "\n### Batch-size sensitivity at N = 64\n", *batch_table(),
           "\n### One-at-a-time sensitivity at N = 64 (LHAS / DP(N) / OWT, ms)\n", *sens_table(),
           "\n### Verification coverage of selected operations\n", *verification_table(rows),
           "\n### Metropolis search accounting (initialization included; review 0892134, F4)\n", *mcmc_table(rows),
           "\n### Measured T4 computation (sensitivity study; analytical values in parentheses)\n", *profiled_table(),
           "\nSelected LHAS configurations:\n"]
    out += [f"* {r['workload']} N={r['N']}: `{r['lhas_configs'][:600]}`" for r in rows]
    (PROC / "tables.md").write_text("\n".join(out) + "\n")
    (PROC / "table_main.tex").write_text("\n".join(tex) + "\n")
    print("wrote", PROC / "tables.md", PROC / "table_main.tex")



if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
