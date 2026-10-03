#!/usr/bin/env python
"""Summaries and figures from the final results (results/raw/final*). Missing or
unresolved runs are reported with their status, never filled.

Outputs:
  results/processed/main_summary.csv        per workload/N: every method's value and status
  results/processed/sensitivity_summary.csv per setting/workload at N = 64
  results/processed/family_summary.csv      collective-family comparison at N = 64
  results/processed/runtime_summary.csv     construction / plan / baseline / verification seconds
  results/processed/status.md               which runs exist and their LHAS status
  figures/*.pdf, figures/*.png
All times are predictions of the analytical model; none is a measurement.
"""
from __future__ import annotations

import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from results_lib import (ROOT, RAW, WORKLOADS, CHAIN, NS, NOTE, load, value, gpipe_value)  # noqa: E402

PROC = ROOT / "results" / "processed"
FIG = ROOT / "figures"
METHODS = [("lhas", "LHAS"), ("dp_N", "DP (p = N)"), ("dp_best", "DP (best common p)"), ("owt", "OWT (adapted)"),
           ("flexflow_mcmc", "FlexFlow-MCMC (adapted)"), ("gpipe", "GPipe (adapted, D13)")]


def main_rows(tag="final"):
    rows, status = [], []
    for w in WORKLOADS:
        for N in NS + ([1024] if w == "alexnet" else []):
            d = load(tag if N != 1024 else "final_supp", w, N)
            if d is None:
                status.append(f"| {w} | {N} | missing | |")
                continue
            bl = d.get("baselines", {})
            row = dict(workload=w, N=N, B=d["B"], supplementary=(N == 1024))
            for k, _ in METHODS:
                if k == "lhas":
                    v, s = value(d["lhas"])
                elif k == "gpipe":
                    if w not in CHAIN:
                        v, s = math.nan, "not applicable"
                    else:
                        v, s, _ = gpipe_value(w, N, "final_gpipe" if N != 1024 else "final_supp_gpipe")
                else:
                    v, s = value(bl.get(k))
                row[k], row[k + "_status"] = v, s
            db = bl.get("dp_best") or {}
            row["dp_best_p"] = db.get("p")
            row["dp_best_unresolved_counts"] = " ".join(map(str, db.get("unresolved_counts") or []))
            sn = bl.get("single_node") or {}
            row["single_node_time_only"] = sn.get("time_only")
            row["single_node_status"] = sn.get("status")
            mc = bl.get("flexflow_mcmc") or {}
            row["mcmc_not_established_evaluations"] = mc.get("not_established_evaluations")
            # per-seed distinct evaluations summed over seeds, initialization included (F4)
            row["mcmc_evaluations"] = sum(r["counts"]["evaluations"] for r in mc.get("runs", []))
            row["mcmc_evaluations_initialization"] = mc.get("evaluations_initialization_sum")
            row["mcmc_evaluations_globally_distinct"] = mc.get("evaluations_globally_distinct")
            row["mcmc_seconds_sum"] = mc.get("seconds_sum")
            row["dp_best_certification"] = db.get("certification")
            lh = d["lhas"]
            row["lhas_configs"] = " ".join(lh.get("configs", []))
            row["lhas_memory_max_GiB"] = (lh.get("memory_max") or math.nan) / 2 ** 30
            vs = (d.get("verification") or {}).get("summary") or {}
            row["verify_expected_ops"] = vs.get("expected_operations")
            row["verify_network_ops"] = vs.get("network_operations")
            row["verify_by_status"] = " ".join(f"{k}:{v}" for k, v in sorted((vs.get("by_status") or {}).items()))
            row["verify_all_checked"] = vs.get("all_network_operations_checked")
            row["verify_packing_checked"] = vs.get("packing_checked")
            rt = d.get("runtime", {})
            row.update(construction_s=rt.get("construction_seconds"), plan_s=rt.get("plan_seconds"),
                       baselines_s=rt.get("baselines_seconds"), verification_s=rt.get("verification_seconds"),
                       total_s=rt.get("total_seconds"), peak_rss_GiB=(rt.get("peak_rss_bytes") or 0) / 2 ** 30,
                       pack_cache_files_at_start=d.get("provenance", {}).get("pack_cache_files_at_start"),
                       code_commit=d.get("provenance", {}).get("code_commit"))
            rows.append(row)
            status.append(f"| {w} | {N} | present | {row['lhas_status']} |")
    return rows, status


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def fig_main(rows):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    styles = dict(lhas="o-", dp_N="s--", dp_best="d-.", owt="^--", flexflow_mcmc="x:", gpipe="v:")
    for ax, w in zip(axes, WORKLOADS):
        rs = [r for r in rows if r["workload"] == w]
        for k, lab in METHODS:
            pts = [(r["N"], r[k] * 1e3) for r in rs if not math.isnan(r[k])]
            if pts:
                ax.plot(*zip(*pts), styles[k], label=lab, ms=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(w + (" (N = 1024 supplementary)" if any(r["N"] == 1024 for r in rs) else ""), fontsize=9)
        ax.set_xlabel("N (nodes)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("predicted time per iteration (ms, log)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, frameon=False)
    fig.suptitle(NOTE + " Unresolved or infeasible entries are omitted (see tables).", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_main_comparison.{ext}", dpi=150)
    plt.close(fig)


def fig_configs(tag="final"):
    for w in CHAIN:
        present = [(N, load(tag, w, N)) for N in NS]
        present = [(N, d) for N, d in present if d and d["lhas"].get("configs")]
        if not present:
            continue
        n = len(present[0][1]["lhas"]["configs"])
        fig, ax = plt.subplots(figsize=(1.2 + 0.5 * n, 0.6 + 0.45 * len(present)))
        for yi, (N, d) in enumerate(present):
            for xi, c in enumerate(d["lhas"]["configs"]):
                col = "#2a9d8f" if c.startswith("DP") else "#e76f51"
                ax.add_patch(plt.Rectangle((xi, yi), 0.95, 0.9, color=col))
                ax.text(xi + 0.47, yi + 0.45, c[2:], ha="center", va="center", fontsize=7, color="white")
        ax.set_xlim(0, n)
        ax.set_ylim(0, len(present))
        ax.set_yticks([i + 0.45 for i in range(len(present))])
        ax.set_yticklabels([f"N={N}" for N, _ in present])
        ax.set_xticks([i + 0.47 for i in range(n)])
        ax.set_xticklabels([f"L{i + 1}" for i in range(n)], fontsize=7)
        ax.set_title(f"{w}: selected configurations (green DP, orange MP; number = p_i)", fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_configs_{w}.{ext}", dpi=150)
        plt.close(fig)


def fig_breakdown(tag="final", N=64):
    cats = ["compute", "intra_transport", "intra_reduction", "inter"]
    labels, vals = [], []
    for w in CHAIN:
        d = load(tag, w, N)
        if not d:
            continue
        for name, e in (("LHAS", d["lhas"]), ("DP", d["baselines"].get("dp_N")), ("OWT", d["baselines"].get("owt"))):
            if e and e.get("breakdown") and e.get("status") == "feasible":
                labels.append(f"{w}\n{name}")
                vals.append([e["breakdown"][c] * 1e3 for c in cats])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 3.4))
    bottom = [0.0] * len(vals)
    for ci, c in enumerate(cats):
        h = [v[ci] for v in vals]
        ax.bar(range(len(vals)), h, bottom=bottom, label=c.replace("_", " "))
        bottom = [b + x for b, x in zip(bottom, h)]
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("ms per iteration")
    ax.set_title(f"Charge breakdown at N={N} (additive model, alpha=0)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_breakdown_N{N}.{ext}", dpi=150)
    plt.close(fig)


def sensitivity_rows(N=64):
    base = RAW / "final_sens"
    rows = []
    if not base.exists():
        return rows
    for tagdir in sorted(base.iterdir()):
        for w in WORKLOADS:
            d = load(f"final_sens/{tagdir.name}", w, N)
            if d is None:
                continue
            bl = d.get("baselines", {})
            r = dict(setting=tagdir.name, workload=w, N=N)
            for k in ("lhas", "dp_N", "owt"):
                r[k], r[k + "_status"] = value(d["lhas"] if k == "lhas" else bl.get(k))
            r["lhas_configs"] = " ".join(d["lhas"].get("configs", []))
            rows.append(r)
    return rows


def fig_sensitivity(rows):
    if not rows:
        return
    groups = [("alpha", ["nominal", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1.0"]),
              ("rho", ["rho_0.25", "nominal", "rho_0.75", "rho_1.0"]),
              ("beta_red eff.", ["redeff_0.25", "nominal", "redeff_1.0"]),
              ("t_setup", ["tsetup_5e-6", "nominal", "tsetup_100e-6"]),
              ("lambda = W", ["lamW_32", "nominal", "lamW_128", "lamW_256"]),
              ("reach", ["nominal", "reach_15", "reach_16", "reach_unrestricted_nopeer"]),
              ("assumptions", ["nominal", "raw_retain", "reserve_0", "no_momentum", "eps_0.5", "whole_item"])]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.3), sharey=False)
    for ax, (g, settings) in zip(axes, groups):
        for w, mk in zip(WORKLOADS, "osd^"):
            nom = next((r["lhas"] for r in rows if r["workload"] == w and r["setting"] == "nominal"), math.nan)
            ys = [next((r["lhas"] for r in rows if r["workload"] == w and r["setting"] == s), math.nan) / nom
                  for s in settings]
            ax.plot(range(len(settings)), ys, mk + "-", label=w, ms=4)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([s.replace("nominal", "nom").split("_", 1)[-1] if g != "assumptions" else s.replace("nominal", "nom")
                            for s in settings], rotation=45, fontsize=7)
        ax.set_title(g, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LHAS time / nominal")
    axes[-1].legend(fontsize=7)
    fig.suptitle("One-at-a-time sensitivity at N=64, B=1024. " + NOTE, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_sensitivity.{ext}", dpi=150)
    plt.close(fig)


def family_rows(N=64):
    rows = []
    for w in WORKLOADS:
        for f in ("all", "one-stage", "deepest"):
            d = load(f"final_family/{f}", w, N)
            v, s = value(d["lhas"]) if d else (math.nan, "missing")
            rows.append(dict(workload=w, N=N, family=f, lhas=v, status=s,
                             structures=len(d.get("family_candidates") or {}) if d else None))
    return rows


def fig_family(rows):
    if not any(r["status"] != "missing" for r in rows):
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fams = ["all", "one-stage", "deepest"]
    for j, f in enumerate(fams):
        xs, ys = [], []
        for i, w in enumerate(WORKLOADS):
            base = next(r["lhas"] for r in rows if r["workload"] == w and r["family"] == "all")
            v = next(r["lhas"] for r in rows if r["workload"] == w and r["family"] == f)
            xs.append(i + (j - 1) * 0.27)
            ys.append(v / base if base == base and v == v else math.nan)
        ax.bar(xs, ys, width=0.25, label=f)
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels(WORKLOADS)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_ylabel("LHAS time / all families")
    ax.set_title("Collective-family comparison at N = 64 (LHAS re-planned per family)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_collective_families.{ext}", dpi=150)
    plt.close(fig)


def fig_mcmc_trace(N=64, tag="final"):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, w in zip(axes, WORKLOADS):
        d = load(tag, w, N)
        if not d:
            continue
        mc = (d.get("baselines") or {}).get("flexflow_mcmc") or {}
        for r in mc.get("runs", []):
            tr = r.get("trace") or []
            if tr:
                ax.step([t[1] for t in tr], [float(t[3]) * 1e3 for t in tr], where="post", label=f"seed {r['seed']}")
        v, s = value(d["lhas"])
        if v == v:
            ax.axhline(v * 1e3, color="k", ls="--", lw=0.8, label="LHAS")
        ax.set_title(f"{w}, N = {N}", fontsize=9)
        ax.set_xlabel("evaluations (including initialization)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("incumbent (ms)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_mcmc_trace_N{N}.{ext}", dpi=150)
    plt.close(fig)


def fig_runtime(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for w, mk in zip(WORKLOADS, "osd^"):
        rs = [r for r in rows if r["workload"] == w and r.get("plan_s") is not None]
        if not rs:
            continue
        axes[0].plot([r["N"] for r in rs], [r["construction_s"] or math.nan for r in rs], mk + "-", label=w)
        axes[1].plot([r["N"] for r in rs], [r["plan_s"] for r in rs], mk + "-", label=w)
    for ax, t in zip(axes, ("collective construction and evaluation (s)", "configuration search (s)")):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.suptitle("Planner runtime on a 2-core container (typed-interface engine: construction inside the search)",
                 fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_planner_runtime.{ext}", dpi=150)
    plt.close(fig)


def fig_construction():
    import json
    f = ROOT / "results" / "pilot" / "allpairs_N1024.jsonl"
    if not f.exists():
        return
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["N"] == 1024]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([r["p"] for r in recs], [r["seconds"] for r in recs], "o-")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("p (all ordered pairs among nodes 1..p)")
    ax.set_ylabel("first-fit packing time (s)")
    ax.set_title("Construction cost, one phase, N=1024\n(2-core container; exact rule)", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_construction_cost.{ext}", dpi=150)
    plt.close(fig)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, status = main_rows()
    write_csv(PROC / "main_summary.csv", rows)
    srows = sensitivity_rows()
    write_csv(PROC / "sensitivity_summary.csv", srows)
    frows = family_rows()
    write_csv(PROC / "family_summary.csv", frows)
    (PROC / "status.md").write_text("| workload | N | main run | LHAS status |\n|---|---|---|---|\n"
                                    + "\n".join(status) + "\n")
    fig_main(rows)
    fig_configs()
    fig_breakdown()
    fig_sensitivity(srows)
    fig_family(frows)
    fig_mcmc_trace()
    fig_runtime(rows)
    fig_construction()
    print(f"{len(rows)} main rows, {len(srows)} sensitivity rows, {len(frows)} family rows -> {PROC}, {FIG}")


if __name__ == "__main__":
    main()
