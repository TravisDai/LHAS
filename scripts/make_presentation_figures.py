"""Redraw presentation figures from frozen run JSON, without running experiments.

Default exports a small, hash-grounded data snapshot and draws figures.
--data reuses that snapshot for standalone figure reproduction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ["alexnet", "vgg16", "googlenet", "resnet50"]
NAMES = dict(zip(WORKLOADS, ["AlexNet", "VGG16", "GoogLeNet", "ResNet50"]))
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#555555"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})


def snapshot():
    sources = {}
    def load(tag, w, n):
        p = ROOT / "results/raw" / tag / f"{w}_N{n}.json"
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    def entry(e):
        status = e.get("status")
        if "best_cost" in e:
            status = "resource-limited" if any(r["counts"].get("not_established", 0) for r in e["runs"]) else "feasible"
            return {"seconds": e["best_cost"], "status": status}
        return {"seconds": e.get("cost") if status in {"feasible", "incumbent"} else None, "status": status}
    out = {"baseline_commit": "e8f8d15", "units": "seconds", "main": [], "ablations": [],
           "families": [], "sensitivity": [], "batch": [], "profiled": []}
    for w in WORKLOADS:
        for n in [64, 128, 256, 512] + ([1024] if w == "alexnet" else []):
            d = load("final_supp" if n == 1024 else "final", w, n)
            row = {"workload": w, "N": n, "lhas": entry(d["lhas"]), "configs": d["lhas"]["configs"],
                   "layers": d["lhas"].get("layers"), "runtime": d["runtime"]}
            for k in ["dp_N", "dp_best", "owt", "flexflow_mcmc"]:
                row[k] = entry(d["baselines"][k])
            if w in ["alexnet", "vgg16"]:
                row["gpipe"] = entry(load("final_supp_gpipe" if n == 1024 else "final_gpipe", w, n)["result"])
            out["main"].append(row)
    for w in ["alexnet", "vgg16"]:
        for n in [64, 256]:
            d = load("final_ablations", w, n)
            out["ablations"].append({"workload": w, "N": n, "lhas": entry(d["lhas"]),
                                     **{k: entry(v) for k, v in d["baselines"]["ablations"].items()}})
    for w in WORKLOADS:
        for f in ["all", "one-stage", "deepest"]:
            d = load("final_family/" + f, w, 64)
            out["families"].append({"workload": w, "family": f, "lhas": entry(d["lhas"])})
        for n in [64, 256]:
            d = load("final_profiled_T4", w, n)
            out["profiled"].append({"workload": w, "N": n, "lhas": entry(d["lhas"]),
                                    **{k: entry(d["baselines"][k]) for k in ["dp_N", "dp_best", "owt"]}})
        for b in [1024, 256]:
            d = load("final" if b == 1024 else "final_batch/B256", w, 64)
            out["batch"].append({"workload": w, "B": b, "lhas": entry(d["lhas"]), "dp_best": entry(d["baselines"]["dp_best"])})
    for folder in sorted((ROOT / "results/raw/final_sens").iterdir()):
        if not folder.is_dir():
            continue
        for w in WORKLOADS:
            d = load("final_sens/" + folder.name, w, 64)
            out["sensitivity"].append({"workload": w, "setting": folder.name, "lhas": entry(d["lhas"]),
                                       **{k: entry(d["baselines"][k]) for k in ["dp_N", "owt"]}})
    out["source_sha256"] = sources
    return out


def seconds(row, k="lhas"):
    e = row[k]
    if e["seconds"] is None:
        return np.nan
    assert e["status"] in {"feasible", "incumbent", "resource-limited"}
    return e["seconds"]


def main_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    methods = ["lhas", "dp_N", "dp_best", "owt", "flexflow_mcmc", "gpipe"]
    labels = ["LHAS", "DP (all nodes)", "DP (best common count)", "OWT (adapted)", "FlexFlow-derived", "GPipe (adapted)"]
    for ax, w, letter in zip(axes.flat, WORKLOADS, "abcd"):
        rs = [r for r in data["main"] if r["workload"] == w]
        for k, c, marker, ls, lab in zip(methods, COLORS, "osD^xv", ["-", "--", "-.", "--", ":", ":"], labels):
            pts = [(r["N"], seconds(r, k) * 1e3) for r in rs if k in r]
            if pts:
                ax.plot(*zip(*pts), color=c, marker=marker, linestyle=ls, label=lab,
                        markersize=4, linewidth=1.2, zorder=5 if k == "lhas" else 3)
        ax.set(xscale="log", yscale="log", title=f"({letter}) {NAMES[w]}", xlabel="Physical nodes, N")
        ax.set_xticks([r["N"] for r in rs], [str(r["N"]) + ("*" if r["N"] == 1024 else "") for r in rs])
        ax.minorticks_off()
        ax.grid(axis="y", alpha=.22)
    for ax in axes[:, 0]: ax.set_ylabel("Predicted time (ms)")
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               columnspacing=1.1, handlelength=1.8)
    fig.tight_layout(rect=(0, .10, 1, 1))
    return fig


def ablations(data):
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.3))
    ax = axes[0]; x = np.arange(4); width = .19
    keys = ["strategy_only", "dp_only_counts", "mp_only_counts", "greedy_layerwise"]
    for j, (k, lab, c, hatch) in enumerate(zip(keys, ["Strategy only", "DP only", "MP only", "Greedy"], [COLORS[0],COLORS[2],COLORS[1],COLORS[3]], ["", "//", "xx", ".."])):
        ys = [seconds(r, k) / seconds(r) for r in data["ablations"]]
        ax.bar(x+(j-1.5)*width, ys, width, color=c, label=lab, hatch=hatch, edgecolor="white", linewidth=.5)
        for i, y in enumerate(ys):
            if not np.isfinite(y):
                assert data["ablations"][i][k]["status"] == "infeasible"
                ax.text(x[i]+(j-1.5)*width, 1.3, "infeasible", rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [NAMES[r["workload"]] + f"\nN = {r['N']}" for r in data["ablations"]])
    ax.set(ylim=(0,9.4), ylabel="Restricted / LHAS", title="(a) Planner ablations")
    ax.legend(ncol=4, loc="upper left", frameon=False, fontsize=8)
    ax = axes[1]
    for j, (f, lab, c, hatch) in enumerate([("one-stage", "One-stage", COLORS[0], ""), ("deepest", "Deepest", COLORS[1], "//")]):
        ys=[]
        for w in WORKLOADS:
            base = next(r for r in data["families"] if r["workload"] == w and r["family"] == "all")
            r = next(r for r in data["families"] if r["workload"] == w and r["family"] == f)
            ys.append(seconds(r) / seconds(base))
        ax.bar(x+(j-.5)*.28, ys, .27, label=lab, color=c, hatch=hatch, edgecolor="white", linewidth=.5)
    ax.set_xticks(x, [NAMES[w] for w in WORKLOADS]);ax.set(ylim=(0,2.65), ylabel="Family / full LHAS", title="(b) Collective families at N = 64")
    ax.legend(ncol=2, loc="upper left", frameon=False)
    for ax in axes:
        ax.axhline(1, color="#555555", linestyle="--", linewidth=.8)
        ax.grid(axis="y", alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def sensitivities(data):
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.5))
    groups = [
        ("(a) Overlap", r"$\alpha$", [0,.25,.5,.75,1], ["nominal", "alpha_0.25", "alpha_0.5", "alpha_0.75", "alpha_1.0"], None),
        ("(b) Compute utilization", r"$\rho$", [.25,.5,.75,1], ["rho_0.25", "nominal", "rho_0.75", "rho_1.0"], None),
        ("(c) Circuit setup", r"$t_{\rm reconf}$ ($\mu$s)", [5,25,100], ["tsetup_5e-6", "nominal", "tsetup_100e-6"], None),
        ("(d) Wavelength / interface budget", r"$\lambda_{\max}=W_{\rm tx}=W_{\rm rx}$", list(range(4)), ["lamW_32", "nominal", "lamW_128", "lamW_256"], ["32","64","128","256"]),
        ("(e) Reach", "Directed segments", list(range(4)), ["nominal", "reach_15", "reach_16", "reach_unrestricted_nopeer"], ["8","15","16","No limits"]),
        ("(f) Effective bandwidth", r"$\varepsilon$", [.5,1], ["eps_0.5", "nominal"], None)]
    for ax, (title, label, xs, settings, ticks) in zip(axes.flat, groups):
        for w, c, mk in zip(WORKLOADS,COLORS,"osD^"):
            base=next(r for r in data["sensitivity"] if r["workload"] == w and r["setting"] == "nominal")
            ys=[seconds(next(r for r in data["sensitivity"] if r["workload"] == w and r["setting"] == tag))/seconds(base) for tag in settings]
            ax.plot(xs,ys,color=c,marker=mk,label=NAMES[w],markersize=3.5,linewidth=1)
        ax.set_title(title, fontsize=9);ax.set_xlabel(label, fontsize=9)
        ax.set_xticks(xs, ticks if ticks else [f"{v:g}" for v in xs]);ax.tick_params(labelsize=8)
        if ticks: ax.tick_params(axis="x",labelrotation=15)
        ax.axhline(1,color="#555555",linestyle="--",linewidth=.7);ax.grid(axis="y",alpha=.18)
    for ax in axes[:,0]: ax.set_ylabel("LHAS / nominal")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.07,1,1))
    return fig


def batch(data):
    fig,ax=plt.subplots(figsize=(6.6,2.35));x=np.arange(4)
    for j,b in enumerate([1024,256]):
        rs=[next(r for r in data["batch"] if r["workload"] == w and r["B"] == b) for w in WORKLOADS]
        ys=[seconds(r,"dp_best")/seconds(r) for r in rs]
        ax.bar(x+(j-.5)*.3,ys,.29,label=f"B = {b}",color=COLORS[j],hatch="//" if j else "",edgecolor="white",linewidth=.5)
    ax.set_xticks(x,[NAMES[w] for w in WORKLOADS]);ax.set(ylabel="DP best / LHAS",ylim=(0,4.9))
    ax.axhline(1,color="#555555",linestyle="--",linewidth=.8);ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    ax.legend(ncol=2,loc="upper right",frameon=False)
    fig.tight_layout();return fig


def configs(data):
    fig,axes=plt.subplots(2,1,figsize=(6.6,3.45))
    for ax,w,letter in zip(axes,["alexnet","vgg16"],"ab"):
        rs=[r for r in data["main"] if r["workload"] == w and r["N"] <= 512]
        n=len(rs[0]["configs"]); conv=n-3
        for y,r in enumerate(rs):
            for x,c in enumerate(r["configs"]):
                mp=c.startswith("MP")
                ax.add_patch(Rectangle((x,y),1,1,facecolor="#D55E00" if mp else "#009E73",edgecolor="white",linewidth=.7,hatch="//" if mp else None))
                ax.text(x+.5,y+.5,c[2:],ha="center",va="center",color="white",fontsize=8,fontweight="bold")
        ax.set(xlim=(0,n),ylim=(4,0),title=f"({letter}) {NAMES[w]}")
        ax.set_yticks(np.arange(4)+.5,[f"N = {r['N']}" for r in rs])
        ax.set_xticks(np.arange(n)+.5,[f"C{i+1}" if i<conv else f"F{i-conv+1}" for i in range(n)],fontsize=8)
        ax.axvline(conv,color="black",linestyle="--",linewidth=.8)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[Patch(facecolor="#009E73",label="DP"),Patch(facecolor="#D55E00",hatch="//",label="MP")],loc="lower center",ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.07,1,1));return fig


def profiled(data):
    fig,axes=plt.subplots(2,1,figsize=(6.6,4.3));x=np.arange(4)
    keys=["dp_N","dp_best","owt"]; labs=["DP (all nodes)","DP (best common count)","OWT"]
    width=.115
    for ax,n,letter in zip(axes,[64,256],"ab"):
        for ki,(k,c) in enumerate(zip(keys,[COLORS[1],COLORS[2],COLORS[3]])):
            for j,mode in enumerate(["profiled","main"]):
                rs=[next(r for r in data[mode] if r["workload"] == w and r["N"] == n) for w in WORKLOADS]
                ys=[seconds(r,k)/seconds(r) for r in rs]
                ax.bar(x+(ki*2+j-2.5)*width,ys,width*.95,color=c if mode=="profiled" else "white",edgecolor=c,
                       hatch=None if mode=="profiled" else "///",linewidth=.7)
        ax.set_xticks(x,[NAMES[w] for w in WORKLOADS]);ax.set(ylabel="Baseline / LHAS",title=f"({letter}) Modeled ring: N = {n}")
        ax.set_ylim(0,4.55 if n==64 else 6.4);ax.axhline(1,color="#555555",linestyle="--",linewidth=.8)
        ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    handles=[Patch(facecolor=c,label=lab) for c,lab in zip([COLORS[1],COLORS[2],COLORS[3]],labs)]
    handles += [Patch(facecolor="#777777",label="T4 compute inputs"),Patch(facecolor="white",edgecolor="#777777",hatch="///",label="Analytical reference")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.1,1,1));return fig


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path);p.add_argument("--output",type=Path,default=ROOT/"figures"/"presentation")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.data.read_text()) if a.data else snapshot()
    if not a.data:
        (a.output/"presentation_plot_data.json").write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    for name,fn in [("fig_main_comparison_compact",main_figure),("fig_ablations_families",ablations),
                    ("fig_sensitivity_compact",sensitivities),("fig_batch_comparison",batch),
                    ("fig_layer_configurations",configs),("fig_profiled_comparison",profiled)]:
        fig=fn(data)
        for ext in ["pdf","png","svg"]:fig.savefig(a.output/f"{name}.{ext}",dpi=180,bbox_inches="tight",pad_inches=.03)
        plt.close(fig)
    print("Drew six figures from",len(data["source_sha256"]),"hash-grounded frozen run files. No experiments run.")

if __name__ == "__main__":main()
