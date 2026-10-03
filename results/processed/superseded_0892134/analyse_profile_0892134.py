#!/usr/bin/env python
"""Analysis of a returned Colab profile (default: the Tesla T4 run of 2026-09-28).

Reads the unmodified ZIP and writes
  results/processed/t4_validation.md       coverage, notebook validation metrics, additional
                                           time-weighted and size-resolved statistics, scaling
                                           efficiency, reduction rate
  figures/fig_t4_compute_validation.{pdf,png}
The notebook's own held-out split (seed 0) is reproduced exactly. These are statistics of
local-computation measurements on one GPU; nothing here concerns the optical model.

Usage: python scripts/analyse_profile.py [zip]
"""
from __future__ import annotations

import collections
import csv
import io
import json
import random
import statistics
import sys
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "results" / "colab" / "Tesla_T4_2026-09-28" / "lhas_profile_Tesla_T4.zip"
PARTS = ("forward", "input_grad", "weight_grad")
T4_FP32_PEAK = 8.1e12          # NVIDIA Tesla T4 datasheet, single precision
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8984"


def load(zpath):
    z = zipfile.ZipFile(zpath)
    names = {Path(n).name: n for n in z.namelist()}
    rd = lambda n: list(csv.DictReader(io.TextIOWrapper(z.open(names[n]), "utf-8")))  # noqa: E731
    js = lambda n: json.loads(z.read(names[n])) if n in names else None  # noqa: E731
    return dict(env=js("environment.json"), cfg=js("config.json"), check=js("manifest_check.json"),
                val=js("device_model_validation.json"), rows=rd("layer_timings_summary.csv"),
                red=rd("reduction_timings.csv") if "reduction_timings.csv" in names else [])


def main(zpath=ZIP):
    d = load(zpath)
    rows = d["rows"]
    meas_rows = [r for r in rows if r["status"] == "measured"]
    tot = lambda r: sum(float(r[f"{k}_median_s"]) for k in PARTS)  # noqa: E731
    fl = np.array([3 * float(r["flops_per_product"]) for r in meas_rows])
    t = np.array([tot(r) for r in meas_rows])
    kind = np.array([r["kind"] for r in meas_rows])
    eff = fl / t
    # the notebook's split (seed 0)
    ids = sorted({int(r["shape_id"]) for r in meas_rows})
    random.Random(d["cfg"]["SEED"]).shuffle(ids)
    cal_ids = set(ids[: len(ids) // 2])
    cal = np.array([int(r["shape_id"]) in cal_ids for r in meas_rows])
    R = fl[cal].sum() / t[cal].sum()
    pred, meas = fl[~cal] / R, t[~cal]
    rel = (pred - meas) / meas
    tw_abs = float(np.abs(pred - meas).sum() / meas.sum())
    # comparison only: affine model t = a + F/R' fitted on the same calibration shapes by least
    # squares on relative residuals (not used by the planner)
    A = np.stack([np.ones(int(cal.sum())), fl[cal]], 1) / t[cal][:, None]
    (a_aff, b_aff), *_ = np.linalg.lstsq(A, np.ones(int(cal.sum())), rcond=None)
    pa = a_aff + b_aff * fl[~cal]
    ea = np.abs(pa - meas) / meas
    affine = dict(a_ms=a_aff * 1e3, R_tflops=1 / b_aff / 1e12, mdare=float(np.median(ea)),
                  p90=float(np.percentile(ea, 90)), tw=float(np.abs(pa - meas).sum() / meas.sum()))
    tw_signed = float((pred.sum() - meas.sum()) / meas.sum())
    # coverage
    uses_all = sum(len(json.loads(r["uses"])) for r in rows)
    uses_meas = sum(len(json.loads(r["uses"])) for r in meas_rows)
    unmeasured = collections.Counter()
    for r in rows:
        if r["status"] != "measured":
            for u in json.loads(r["uses"]):
                unmeasured[(u["workload"], u["layer"])] += 1
    # parts above the FP32 peak (direct-convolution FLOP count)
    above = collections.Counter()
    for r in meas_rows:
        for k in PARTS:
            if float(r["flops_per_product"]) / float(r[f"{k}_median_s"]) > T4_FP32_PEAK:
                above[(r["kind"], r["kernel"], r["stride"])] += 1
    # scaling efficiency of DP counts relative to the smallest measured count, by local batch
    tab = collections.defaultdict(dict)
    for r in meas_rows:
        for u in json.loads(r["uses"]):
            tab[(u["workload"], u["layer"], u["strategy"])][u["p"]] = (tot(r), int(r["batch_local"]))
    scale = collections.defaultdict(list)
    for (w, layer, s), dd in tab.items():
        if s != "DP":
            continue
        p0 = min(dd)
        t0 = dd[p0][0]
        for p, (tp, b) in dd.items():
            scale[b].append(t0 * p0 / (tp * p))
    # reduction
    red = [float(r["model_equivalent_rate_byte_s"]) for r in d["red"]
           if r.get("method") == "fused_sum" and int(r["V_bytes"]) >= 16 * 2 ** 20]
    red_rate = statistics.median(red) if red else None
    bins = [1e6, 1e8, 1e9, 1e10, 1e11, 1e12, 1e14]
    lines = ["# Tesla T4 compute profile: coverage and model validation\n",
             f"Source: `{Path(zpath).relative_to(ROOT) if Path(zpath).is_relative_to(ROOT) else zpath}` "
             f"(notebook v{d['cfg']['notebook_version']}); GPU {d['env']['gpu_name']}, CC {d['env']['compute_capability']}, "
             f"driver {d['env']['driver_version']}, CUDA {d['env']['cuda_runtime_torch']}, cuDNN {d['env']['cudnn_version']}, "
             f"torch {d['env']['torch']}, torchvision {d['env']['torchvision']}; float32, TF32 off; "
             f"manifest check: {d['check']['status']}. Local computation only; no optical measurement.\n",
             "## Coverage\n",
             f"* Unique local shapes: {len(rows)}; measured {len(meas_rows)} "
             f"({100 * len(meas_rows) / len(rows):.1f}%); not measured (memory cap): {len(rows) - len(meas_rows)}.",
             f"* Layer/configuration entries: {uses_meas} measured of {uses_all}.",
             "* Unmeasured entries (all VGG16 early CONV layers at large local batch or full-batch MP): "
             + ", ".join(f"{w}/{layer} ({n})" for (w, layer), n in sorted(unmeasured.items())) + ".",
             f"* Timing repeatability: median IQR/median of the forward product "
             f"{np.median([(float(r['forward_p75_s']) - float(r['forward_p25_s'])) / float(r['forward_median_s']) for r in meas_rows]):.3f}.",
             f"* Total measured time (sum of medians over shapes): {t.sum():.1f} s.\n",
             "## Constant-throughput model on held-out shapes (notebook split, seed 0)\n",
             f"Fitted rate R = {R / 1e12:.3f} TFLOP/s ({100 * R / T4_FP32_PEAK:.0f}% of the {T4_FP32_PEAK / 1e12:.1f} TFLOP/s "
             f"FP32 peak); {int(cal.sum())} calibration and {int((~cal).sum())} validation shapes.\n",
             "| metric | all | CONV | FC |", "|---|---|---|---|"]
    v = d["val"] or {}
    def g(sec, k):  # noqa: E306
        x = (v.get(sec) or {}).get(k)
        return "—" if x is None else f"{x:.3f}"
    for k, lab in (("signed_relative_error_median", "signed relative error, median"),
                   ("absolute_relative_error_median", "absolute relative error, median"),
                   ("absolute_relative_error_p90", "absolute relative error, 90th percentile"),
                   ("absolute_relative_error_p95", "absolute relative error, 95th percentile"),
                   ("absolute_relative_error_max", "absolute relative error, maximum"),
                   ("spearman_rank_correlation", "Spearman rank correlation")):
        lines.append(f"| {lab} | {g('validation', k)} | {g('validation_conv', k)} | {g('validation_fc', k)} |")
    lines += ["",
              f"Additional (computed here from the same split): time-weighted absolute relative error "
              f"sum|pred − meas| / sum(meas) = {tw_abs:.3f}; error of the summed validation time "
              f"(pred − meas)/meas = {tw_signed:+.3f}.\n",
              f"For comparison only, an affine model t = a + F/R' fitted on the same calibration shapes "
              f"(a = {affine['a_ms']:.3f} ms per layer shape, R' = {affine['R_tflops']:.2f} TFLOP/s) gives a median "
              f"absolute relative error of {affine['mdare']:.3f} (90th percentile {affine['p90']:.3f}; time-weighted "
              f"{affine['tw']:.3f}). The remaining spread reflects shape-dependent kernel efficiency.\n",
              "## Achieved rate by shape size (direct-convolution FLOP count, all measured shapes)\n",
              "| 3 × FLOPs per product | shapes | median rate (TFLOP/s) | 10th–90th percentile | share of measured time |",
              "|---|---|---|---|---|"]
    for a, b in zip(bins, bins[1:]):
        m = (fl >= a) & (fl < b)
        if m.any():
            lines.append(f"| {a:.0e}–{b:.0e} | {int(m.sum())} | {np.median(eff[m]) / 1e12:.2f} | "
                         f"{np.percentile(eff[m], 10) / 1e12:.2f}–{np.percentile(eff[m], 90) / 1e12:.2f} | "
                         f"{t[m].sum() / t.sum():.3f} |")
    lines += ["",
              f"{sum(above.values())} individual products exceed the FP32 peak when counted as direct convolutions, all "
              f"stride-1 3×3 or 5×5 convolutions ({', '.join(f'{k[1]}: {n}' for k, n in above.most_common())}); this is "
              "consistent with cuDNN selecting Winograd or FFT algorithms, so direct-convolution FLOP counts overstate the "
              "arithmetic actually performed for these shapes.\n",
              "## DP scaling efficiency by local batch\n",
              "Efficiency = T(p_min)·p_min / (T(p)·p) for each layer's DP configurations (1 = linear speedup).\n",
              "| local batch | layer configurations | median efficiency | 10th–90th percentile |", "|---|---|---|---|"]
    for b in sorted(scale, reverse=True):
        x = np.array(scale[b])
        lines.append(f"| {b} | {len(x)} | {np.median(x):.2f} | {np.percentile(x, 10):.2f}–{np.percentile(x, 90):.2f} |")
    if red_rate:
        lines += ["", "## Local reduction\n",
                  f"Median model-equivalent rate (a+2)V/t of the fused sum for V ≥ 16 MiB: {red_rate / 1e9:.1f} GB/s "
                  f"({len(red)} measurements). This is the rate at which the planner's reduction term reproduces the "
                  "measured time, not a DRAM bandwidth measurement; it is used as β_red in the profiled planning "
                  "sensitivity."]
    out = ROOT / "results" / "processed" / "t4_validation.md"
    out.write_text("\n".join(lines) + "\n")
    summary = dict(R=R, affine=affine, tw_abs=tw_abs, tw_signed=tw_signed, red_rate=red_rate, measured=len(meas_rows), total=len(rows),
                   uses_meas=uses_meas, uses_all=uses_all, above_peak=sum(above.values()),
                   scale={int(b): float(np.median(x)) for b, x in scale.items()}, validation=v)
    (ROOT / "results" / "processed" / "t4_validation.json").write_text(json.dumps(summary, indent=1))
    # figure: (a) achieved rate vs size, (b) predicted vs measured on held-out shapes
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for k, col, lab in (("conv", BLUE, "CONV"), ("fc", ORANGE, "FC")):
        m = kind == k
        ax[0].scatter(fl[m], eff[m] / 1e12, s=9, color=col, alpha=0.55, linewidths=0, label=lab)
    ax[0].axhline(T4_FP32_PEAK / 1e12, color=MUTED, lw=1, ls="--")
    ax[0].text(fl.min() * 1.3, T4_FP32_PEAK / 1e12 * 1.08, "FP32 peak", color=MUTED, fontsize=8)
    ax[0].axhline(R / 1e12, color=INK, lw=1)
    ax[0].text(fl.min() * 1.3, R / 1e12 * 1.08, f"fitted R = {R / 1e12:.2f}", color=INK, fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("FLOPs per iteration of the layer shape (3 products)")
    ax[0].set_ylabel("achieved rate (TFLOP/s)")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    kv = kind[~cal]
    for k, col, lab in (("conv", BLUE, "CONV"), ("fc", ORANGE, "FC")):
        m = kv == k
        ax[1].scatter(meas[m] * 1e3, pred[m] * 1e3, s=9, color=col, alpha=0.55, linewidths=0, label=lab)
    lo, hi = meas.min() * 1e3 * 0.7, meas.max() * 1e3 * 1.4
    ax[1].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("measured time (ms), held-out shapes")
    ax[1].set_ylabel("predicted time (ms)")
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    for a in ax:
        a.grid(alpha=0.25, which="major")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_title("(a) achieved rate by shape size", fontsize=9, loc="left")
    ax[1].set_title("(b) constant-throughput model, held-out shapes", fontsize=9, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig_t4_compute_validation.{ext}", dpi=160)
    plt.close(fig)
    print(out.read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ZIP)
