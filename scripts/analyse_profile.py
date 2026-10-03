#!/usr/bin/env python
"""Analysis of a returned Colab profile (default: the Tesla T4 run of 2026-09-28).

Protocol: docs/t4_validation_protocol.md (declared before the statistics were computed).
The unit of analysis is the timed kernel signature (SIGNATURE_FIELDS); records with the
same signature are one group, each signature counts once, and calibration and validation
share no signature.

Reads the unmodified ZIP and writes
  results/processed/t4_validation.json   every number (the only source for the report,
                                         the figure and the manuscript tokens)
  results/processed/t4_validation.md     report generated from the JSON
  figures/fig_t4_compute_validation.{pdf,png}   generated from the JSON
These are statistics of local-computation measurements on one GPU; nothing here concerns
the optical model or distributed execution.

Usage: python scripts/analyse_profile.py [zip]
"""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import math
import random
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))           # lhas (planner rules for the ranking inputs)
ZIP = ROOT / "results" / "colab" / "Tesla_T4_2026-09-28" / "lhas_profile_Tesla_T4.zip"
OUT_JSON = ROOT / "results" / "processed" / "t4_validation.json"
OUT_MD = ROOT / "results" / "processed" / "t4_validation.md"
FIG = ROOT / "figures" / "fig_t4_compute_validation"
PARTS = ("forward", "input_grad", "weight_grad")
T4_FP32_PEAK = 8.1e12          # NVIDIA Tesla T4 datasheet, single precision (theoretical peak)
SPLIT_SEEDS = range(100)

# ---- protocol (docs/t4_validation_protocol.md) -------------------------------------------
SIGNATURE_FIELDS = ("kind", "x", "n_out_local", "batch_local", "kernel", "stride", "padding", "bias")
PROTOCOL = dict(
    unit="timed kernel signature " + str(SIGNATURE_FIELDS) + "; n_out_full excluded (does not enter the timed "
         "kernels), kept as per-use metadata; device, float32 and TF32-off are constant and asserted",
    records="measured rows of layer_timings_summary.csv, last row per shape_id (ingestion rule)",
    record_time="sum of forward, input-gradient and weight-gradient median times",
    signature_time="median over the signature's records of the record time",
    weighting="each signature once",
    split="signatures sorted by typed value, shuffled with random.Random(seed), first floor(n/2) calibration",
    seed=0,
    estimator="R = sum(3F)/sum(T) over calibration signatures; prediction 3F/R",
    variability="seeds 0-99, median and 5th-95th percentiles over seeds",
    ranking="within (workload, layer), pairs of measured configurations with the nominal planner's "
            "per-configuration time and FLOPs (raw-input gradient omitted for layers consuming the graph input); "
            "split-independent",
    scaling="T(p0) p0 / (T(p) p), p0 = smallest measured DP count of the layer, planner times as for ranking",
    percentiles="linear interpolation between order statistics",
)


def _typed(r: dict, f: str):
    v = r[f]
    if f in ("x", "kernel", "stride", "padding"):
        return tuple(int(t) for t in json.loads(v))
    if f == "bias":
        return v in ("True", "true", "1", True)
    if f == "kind":
        return v
    return int(v)


def kernel_signature(r: dict) -> tuple:
    return tuple(_typed(r, f) for f in SIGNATURE_FIELDS)


def record_time(r: dict) -> float:
    return sum(float(r[f"{k}_median_s"]) for k in PARTS)


def dedupe_resumed(rows: list) -> list:
    """Keep the last row per shape_id (a resumed notebook appends a new row for a shape an
    earlier pass left unmeasured); the rule of scripts/ingest_colab.py."""
    last = {}
    for r in rows:
        last[r["shape_id"]] = r
    return list(last.values())


def load(zpath=ZIP) -> dict:
    z = zipfile.ZipFile(zpath)
    names = {Path(n).name: n for n in z.namelist()}
    rd = lambda n: list(csv.DictReader(io.TextIOWrapper(z.open(names[n]), "utf-8")))  # noqa: E731
    js = lambda n: json.loads(z.read(names[n])) if n in names else None  # noqa: E731
    rows = rd("layer_timings_summary.csv")
    return dict(env=js("environment.json"), cfg=js("config.json"), check=js("manifest_check.json"),
                notebook_validation=js("device_model_validation.json"), rows_raw_count=len(rows),
                rows=dedupe_resumed(rows), raw=rd("layer_timings_raw.csv"),
                red=rd("reduction_timings.csv") if "reduction_timings.csv" in names else [],
                sha256=hashlib.sha256(Path(zpath).read_bytes()).hexdigest())


def group_by_signature(rows: list) -> dict:
    g = collections.defaultdict(list)
    for r in rows:
        g[kernel_signature(r)].append(r)
    return dict(g)


def split_groups(signatures: list, seed: int = 0):
    """Deterministic grouped split. Returns (calibration, validation) signature lists."""
    s = sorted(signatures)
    random.Random(seed).shuffle(s)
    k = len(s) // 2
    cal, val = s[:k], s[k:]
    assert not set(cal) & set(val)
    return cal, val


# ---- statistics ---------------------------------------------------------------------------
def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    out = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return out


def spearman(a, b) -> float:
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def error_stats(pred, meas) -> dict:
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    rel = (pred - meas) / meas
    ab = np.abs(rel)
    return dict(n=int(len(rel)), abs_rel_median=float(np.median(ab)), abs_rel_p90=float(np.percentile(ab, 90)),
                abs_rel_p95=float(np.percentile(ab, 95)), abs_rel_max=float(ab.max()),
                signed_rel_median=float(np.median(rel)), fraction_overpredicted=float((rel > 0).mean()),
                weighted_abs=float(np.abs(pred - meas).sum() / meas.sum()),
                summed_error=float((pred.sum() - meas.sum()) / meas.sum()),
                spearman=spearman(pred, meas))


def _sig_table(groups: dict, sigs: list):
    F = np.array([3.0 * float(groups[s][0]["flops_per_product"]) for s in sigs])
    T = np.array([float(np.median([record_time(r) for r in groups[s]])) for s in sigs])
    kind = [s[0] for s in sigs]
    return F, T, kind


def grouped_validation(groups: dict, seed: int) -> dict:
    cal, val = split_groups(list(groups), seed)
    Fc, Tc, _ = _sig_table(groups, cal)
    Fv, Tv, kv = _sig_table(groups, val)
    R = Fc.sum() / Tc.sum()
    pred = Fv / R
    out = dict(seed=seed, n_calibration=len(cal), n_validation=len(val),
               n_calibration_records=sum(len(groups[s]) for s in cal),
               n_validation_records=sum(len(groups[s]) for s in val), R=float(R),
               all=error_stats(pred, Tv))
    kv = np.array(kv)
    for k in ("conv", "fc"):
        m = kv == k
        out[k] = error_stats(pred[m], Tv[m]) if m.sum() >= 2 else dict(n=int(m.sum()))
    return out, (cal, val, Fc, Tc, Fv, Tv, kv, pred)


def _fit_models(Fc, Tc, Fv, Tv) -> list:
    """Zero-intercept and affine models on the same calibration signatures; the aggregate
    estimator and least squares of relative residuals (same criterion for both forms)."""
    out = []
    R = Fc.sum() / Tc.sum()
    out.append(dict(model="zero-intercept", estimator="aggregate ratio sum(3F)/sum(T) (primary)",
                    intercept_ms=0.0, rate_tflops=R / 1e12, **_short(error_stats(Fv / R, Tv))))
    q = Fc / Tc
    b = q.sum() / (q * q).sum()
    out.append(dict(model="zero-intercept", estimator="least squares of relative residuals",
                    intercept_ms=0.0, rate_tflops=1 / b / 1e12, **_short(error_stats(b * Fv, Tv))))
    A = np.stack([1 / Tc, Fc / Tc], 1)
    (a, b2), *_ = np.linalg.lstsq(A, np.ones(len(Tc)), rcond=None)
    out.append(dict(model="affine a + 3F/R'", estimator="least squares of relative residuals",
                    intercept_ms=a * 1e3, rate_tflops=1 / b2 / 1e12, **_short(error_stats(a + b2 * Fv, Tv))))
    return out


def _short(e):
    return {k: e[k] for k in ("abs_rel_median", "abs_rel_p90", "abs_rel_p95", "weighted_abs", "summed_error")}


def _needs_input_grad():
    """The nominal planner's raw-input-gradient policy (lhas.model.chain.ChainModel.needs_input_grad):
    the input gradient is omitted only for layers that consume the raw graph input when
    raw_input_grad is 'omit' (configs/nominal_experiment.json)."""
    from lhas.model.workload import load_workload
    omit = json.loads((ROOT / "configs" / "nominal_experiment.json").read_text())["raw_input_grad"] == "omit"
    cache = {}

    def need(workload: str, layer: str) -> bool:
        if workload not in cache:
            cache[workload] = load_workload(workload)
        return not (omit and cache[workload].layer(layer).input_is_graph_input)
    return need


def _config_times(rows):
    """(workload, layer) -> {(strategy, p): (time, local FLOPs)} as read by the nominal planner:
    forward and weight-gradient products, plus the input-gradient product unless the planner
    omits it (review ac66b9e, A1). The held-out kernel validation (record_time) keeps all
    three products."""
    need = _needs_input_grad()
    tab = collections.defaultdict(dict)
    for r in rows:
        for u in json.loads(r["uses"]):
            dx = need(u["workload"], u["layer"])
            t = (float(r["forward_median_s"]) + float(r["weight_grad_median_s"])
                 + (float(r["input_grad_median_s"]) if dx else 0.0))
            f = (3.0 if dx else 2.0) * float(r["flops_per_product"])
            tab[(u["workload"], u["layer"])][(u["strategy"], int(u["p"]))] = (t, f)
    return tab


def configuration_ranking(rows) -> dict:
    tab = _config_times(rows)
    distinct = dict(pairs=0, concordant=0, measured_ties=0)
    consecutive = dict(pairs=0, concordant=0, measured_ties=0)
    equal_flop_ratios, eq_p, dp_faster, mp_faster = [], [], 0, 0
    by_wl = collections.defaultdict(lambda: dict(pairs=0, concordant=0))
    layers = 0
    for (w, layer), d in tab.items():
        items = sorted(d.items())
        if len(items) < 2:
            continue
        layers += 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ca, (ta, fa)), (cb, (tb, fb)) = items[i], items[j]
                if fa == fb:
                    if ca[0] != cb[0]:
                        dp, mp = (ta, tb) if ca[0] == "DP" else (tb, ta)
                        equal_flop_ratios.append(max(dp, mp) / min(dp, mp))
                        eq_p.append(ca[1])
                        dp_faster += dp < mp
                        mp_faster += mp < dp
                    continue
                conc = (fa - fb) * (ta - tb) > 0
                distinct["pairs"] += 1
                distinct["concordant"] += conc
                distinct["measured_ties"] += ta == tb
                by_wl[w]["pairs"] += 1
                by_wl[w]["concordant"] += conc
        for s in ("DP", "MP"):
            ps = sorted(p for (st, p) in d if st == s)
            for p, q in zip(ps, ps[1:]):
                (ta, fa), (tb, fb) = d[(s, p)], d[(s, q)]
                if fa == fb:
                    continue
                consecutive["pairs"] += 1
                consecutive["concordant"] += (fa - fb) * (ta - tb) > 0
                consecutive["measured_ties"] += ta == tb
    for x in (distinct, consecutive):
        x["fraction_concordant"] = x["concordant"] / x["pairs"] if x["pairs"] else None
    r = np.array(equal_flop_ratios)
    gt1 = r[np.array(eq_p) > 1] if len(r) else r
    return dict(layers_with_two_or_more_configurations=layers, distinct_flop_pairs=distinct,
                consecutive_count_pairs=consecutive,
                by_workload={w: dict(v, fraction_concordant=v["concordant"] / v["pairs"]) for w, v in sorted(by_wl.items())},
                equal_flop_dp_mp_pairs=dict(n=int(len(r)), ratio_median=float(np.median(r)) if len(r) else None,
                                            ratio_p90=float(np.percentile(r, 90)) if len(r) else None,
                                            fraction_ratio_above_1_25=float((r > 1.25).mean()) if len(r) else None,
                                            fraction_dp_faster=dp_faster / len(r) if len(r) else None,
                                            n_dp_faster=int(dp_faster), n_mp_faster=int(mp_faster),
                                            # DP(1) and MP(1) time the same kernel (ratio 1 by construction)
                                            n_p1_same_kernel=int(sum(1 for q in eq_p if q == 1)),
                                            p_above_1=dict(n=int(len(gt1)),
                                                           ratio_median=float(np.median(gt1)) if len(gt1) else None,
                                                           ratio_p90=float(np.percentile(gt1, 90)) if len(gt1) else None,
                                                           n_dp_faster=int(dp_faster))))


def scaling_efficiency(rows) -> dict:
    tab = _config_times(rows)
    by_batch = collections.defaultdict(list)
    p0_above_1 = []
    for (w, layer), d in sorted(tab.items()):
        dp = {p: t for (s, p), (t, _) in d.items() if s == "DP"}
        if not dp:
            continue
        p0 = min(dp)
        if p0 > 1:
            p0_above_1.append(dict(workload=w, layer=layer, p0=p0))
        for p, t in dp.items():
            by_batch[1024 // p].append(dp[p0] * p0 / (t * p))
    return dict(reference="smallest measured DP count p0 of each layer",
                layers_with_p0_above_1=p0_above_1,
                by_local_batch={int(b): dict(n=len(x), median=float(np.median(x)), p10=float(np.percentile(x, 10)),
                                             p90=float(np.percentile(x, 90)))
                                for b, x in sorted(by_batch.items(), reverse=True)})


def repeatability(rows) -> dict:
    out = {}
    for k in PARTS:
        v = np.array([(float(r[f"{k}_p75_s"]) - float(r[f"{k}_p25_s"])) / float(r[f"{k}_median_s"]) for r in rows])
        out[k] = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))
    return dict(statistic="IQR/median of the repetitions of one record and product, then median and 90th "
                          "percentile over measured records", scope="repetitions within one session on one GPU "
                          "(30 after 10 warm-up iterations); not run-to-run or device-to-device variation", **out)


def raw_sample_checks(d, rows) -> dict:
    samples = collections.defaultdict(list)
    for r in d["raw"]:
        samples[(r["shape_id"], r["kernel_part"])].append(float(r["seconds"]))
    checks, bad = 0, []
    for r in rows:
        for k in PARTS:
            a = np.array(samples[(r["shape_id"], k)])
            for key, v in (("n", len(a)), ("median_s", np.median(a)), ("p25_s", np.percentile(a, 25)),
                           ("p75_s", np.percentile(a, 75))):
                checks += 1
                if not math.isclose(float(r[f"{k}_{key}"]), float(v), rel_tol=1e-10, abs_tol=1e-15):
                    bad.append((r["shape_id"], k, key))
    return dict(checks=checks, mismatches=len(bad), examples=bad[:5])


def notebook_split_superseded(d, meas, groups) -> dict:
    """The notebook's record-ID split, recomputed from the CSV to document the leakage (T1)."""
    ids = sorted(int(r["shape_id"]) for r in meas)
    random.Random(d["cfg"]["SEED"]).shuffle(ids)
    cal_ids = set(ids[: len(ids) // 2])
    leak = [g for g in groups.values() if len({int(r["shape_id"]) in cal_ids for r in g}) == 2]
    F = np.array([3.0 * float(r["flops_per_product"]) for r in meas])
    T = np.array([record_time(r) for r in meas])
    cal = np.array([int(r["shape_id"]) in cal_ids for r in meas])
    R = F[cal].sum() / T[cal].sum()
    e = error_stats(F[~cal] / R, T[~cal])
    nv = d.get("notebook_validation") or {}
    same = all(math.isclose(e[a], (nv.get("validation") or {}).get(b, float("nan")), rel_tol=1e-9)
               for a, b in (("abs_rel_median", "absolute_relative_error_median"),
                            ("abs_rel_p95", "absolute_relative_error_p95"),
                            ("spearman", "spearman_rank_correlation")))
    return dict(status="superseded: calibration and validation share kernel signatures", R=float(R),
                n_calibration=int(cal.sum()), n_validation=int((~cal).sum()), leaking_signatures=len(leak),
                validation_records_with_calibration_twin=sum(sum(int(r["shape_id"]) not in cal_ids for r in g)
                                                             for g in leak),
                validation=e, agrees_with_notebook_json=same)


def rate_by_size(F, T, above_peak_parts) -> list:
    bins = [1e6, 1e8, 1e9, 1e10, 1e11, 1e12, 1e14]
    eff = F / T
    out = []
    for a, b in zip(bins, bins[1:]):
        m = (F >= a) & (F < b)
        if m.any():
            out.append(dict(lo=a, hi=b, n=int(m.sum()), median_tflops=float(np.median(eff[m]) / 1e12),
                            p10_tflops=float(np.percentile(eff[m], 10) / 1e12),
                            p90_tflops=float(np.percentile(eff[m], 90) / 1e12),
                            share_of_time=float(T[m].sum() / T.sum())))
    return out


def analyse(zpath=ZIP) -> dict:
    d = load(zpath)
    env = d["env"]
    assert env.get("precision") == "float32" and not env.get("allow_tf32_matmul") and not env.get("allow_tf32_cudnn")
    rows = d["rows"]
    meas = [r for r in rows if r["status"] == "measured"]
    groups = group_by_signature(meas)
    all_groups = group_by_signature(rows)
    # primary split and its variability
    primary, (cal, val, Fc, Tc, Fv, Tv, kv, pred) = grouped_validation(groups, PROTOCOL["seed"])
    val_ids = {r["shape_id"] for s in val for r in groups[s]}
    cal_ids = {r["shape_id"] for s in cal for r in groups[s]}
    assert not ({kernel_signature(r) for r in meas if r["shape_id"] in val_ids}
                & {kernel_signature(r) for r in meas if r["shape_id"] in cal_ids})
    seeds = [grouped_validation(groups, s)[0] for s in SPLIT_SEEDS]
    var = {}
    for k in ("abs_rel_median", "abs_rel_p90", "abs_rel_p95", "abs_rel_max", "signed_rel_median", "weighted_abs",
              "summed_error", "spearman"):
        x = np.array([s["all"][k] for s in seeds])
        var[k] = dict(median=float(np.median(x)), p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                      seed0_percentile=float((x < primary["all"][k]).mean() * 100))
    x = np.array([s["R"] for s in seeds])
    var["R"] = dict(median=float(np.median(x)), p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))
    # all signatures (split-independent quantities)
    sigs = sorted(groups)
    F, T, kind = _sig_table(groups, sigs)
    above = collections.Counter()
    for s in sigs:
        for k in PARTS:
            t = float(np.median([float(r[f"{k}_median_s"]) for r in groups[s]]))
            if float(groups[s][0]["flops_per_product"]) / t > T4_FP32_PEAK:
                above[f"{s[0]} {s[4][0]}x{s[4][1] if len(s[4]) > 1 else ''} stride {s[5][0] if s[5] else ''}"] += 1
    uses_all = sum(len(json.loads(r["uses"])) for r in rows)
    uses_meas = sum(len(json.loads(r["uses"])) for r in meas)
    unmeasured = collections.Counter()
    for r in rows:
        if r["status"] != "measured":
            for u in json.loads(r["uses"]):
                unmeasured[f"{u['workload']}/{u['layer']}"] += 1
    red = [float(r["model_equivalent_rate_byte_s"]) for r in d["red"]
           if r.get("method") == "fused_sum" and int(r["V_bytes"]) >= 16 * 2 ** 20]
    # reviewer's illustrative variant (string-typed signatures), reproduced only as a cross-check
    sgroups = collections.defaultdict(list)
    for r in meas:
        sgroups[tuple(r[f] for f in SIGNATURE_FIELDS)].append(r)
    ss = sorted(sgroups)
    random.Random(0).shuffle(ss)
    sc, sv = ss[: len(ss) // 2], ss[len(ss) // 2:]
    tab = lambda gs: (np.array([3.0 * float(sgroups[s][0]["flops_per_product"]) for s in gs]),  # noqa: E731
                      np.array([float(np.median([record_time(r) for r in sgroups[s]])) for s in gs]))
    (a1, b1), (a2, b2) = tab(sc), tab(sv)
    res = dict(
        source=dict(zip=str(Path(zpath).relative_to(ROOT)) if Path(zpath).is_relative_to(ROOT) else str(zpath),
                    zip_sha256=d["sha256"], notebook_version=d["cfg"]["notebook_version"],
                    gpu=env["gpu_name"], compute_capability=env["compute_capability"], driver=env["driver_version"],
                    cuda=env["cuda_runtime_torch"], cudnn=env["cudnn_version"], torch=env["torch"],
                    torchvision=env["torchvision"], precision="float32, TF32 off",
                    manifest_check=d["check"]["status"], fp32_peak_flops=T4_FP32_PEAK,
                    fp32_peak_source="NVIDIA Tesla T4 datasheet (theoretical)"),
        protocol=PROTOCOL,
        coverage=dict(summary_rows=d["rows_raw_count"], records=len(rows), measured_records=len(meas),
                      unmeasured_records=len(rows) - len(meas), repeated_ids_removed=d["rows_raw_count"] - len(rows),
                      signatures_all_records=len(all_groups), signatures_measured=len(groups),
                      duplicate_signature_groups=sum(len(g) > 1 for g in groups.values()),
                      records_in_duplicate_groups=sum(len(g) for g in groups.values() if len(g) > 1),
                      uses_all=uses_all, uses_measured=uses_meas, unmeasured_uses=dict(sorted(unmeasured.items())),
                      measured_time_sum_s=float(T.sum())),
        raw_sample_checks=raw_sample_checks(d, meas),
        validation=primary, validation_seed_variability=var,
        spearman_all_signatures=spearman(F, T),
        model_comparison=_fit_models(Fc, Tc, Fv, Tv),
        configuration_ranking=configuration_ranking(meas),
        scaling_efficiency=scaling_efficiency(meas),
        repeatability=repeatability(meas),
        rate_by_size=rate_by_size(F, T, above),
        above_peak=dict(signature_products=sum(above.values()), by_shape=dict(above.most_common()),
                        note="This is consistent with cuDNN selecting Winograd or FFT algorithms; the algorithms were "
                             "not identified by measurement"),
        reduction=dict(median_rate_byte_s=float(np.median(red)) if red else None, n=len(red),
                       definition="median model-equivalent rate (a+2)V/t of the fused sum, V >= 16 MiB"),
        superseded_notebook_split=notebook_split_superseded(d, meas, groups),
        cross_check_string_typed_signatures=dict(
            note="reviewer's illustrative split (signatures compared as CSV strings); implementation cross-check "
                 "only", n_validation=len(sv), abs_rel_median=error_stats(a2 / (a1.sum() / b1.sum()), b2)["abs_rel_median"]),
        figure=dict(all_F=F.tolist(), all_T=T.tolist(), all_kind=kind, val_T=Tv.tolist(), val_pred=pred.tolist(),
                    val_kind=kv.tolist(), R=primary["R"]),
    )
    return res


# ---- report and figure (from the JSON only) ----------------------------------------------
def pct(x, nd=1):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def write_markdown(res: dict, out=OUT_MD):
    s, c, v = res["source"], res["coverage"], res["validation"]
    L = ["# Tesla T4 compute profile: coverage and constant-throughput model validation\n",
         f"Source: `{s['zip']}` (sha256 {s['zip_sha256'][:16]}…; notebook v{s['notebook_version']}); GPU {s['gpu']}, "
         f"CC {s['compute_capability']}, driver {s['driver']}, CUDA {s['cuda']}, cuDNN {s['cudnn']}, torch {s['torch']}, "
         f"torchvision {s['torchvision']}; {s['precision']}; manifest check: {s['manifest_check']}. Local computation "
         "only; no optical, multi-GPU or iteration-time measurement. Protocol: `docs/t4_validation_protocol.md`.\n",
         "## Coverage\n",
         f"* Profiled records (shape IDs): {c['records']}; measured {c['measured_records']}; not measured (memory cap): "
         f"{c['unmeasured_records']}; repeated IDs removed: {c['repeated_ids_removed']}.",
         f"* Distinct timed kernel signatures: {c['signatures_measured']} measured ({c['signatures_all_records']} over "
         f"all records). {c['duplicate_signature_groups']} signatures were timed more than once "
         f"({c['records_in_duplicate_groups']} records), because the notebook keyed records by the parent width too.",
         f"* Layer/configuration entries: {c['uses_measured']} measured of {c['uses_all']}. Unmeasured: "
         + ", ".join(f"{k} ({n})" for k, n in c["unmeasured_uses"].items()) + ".",
         f"* Raw-sample consistency: {res['raw_sample_checks']['checks']} summary values recomputed from the raw "
         f"samples, {res['raw_sample_checks']['mismatches']} mismatches.",
         "* Repeatability, " + res["repeatability"]["scope"] + ". Median IQR/median "
         + ", ".join(f"{k} {pct(res['repeatability'][k]['median'])} (p90 {pct(res['repeatability'][k]['p90'])})"
                     for k in PARTS) + ".\n",
         "## Constant-throughput model, grouped split (no shared signatures)\n",
         f"Seed {v['seed']}: {v['n_calibration']} calibration signatures ({v['n_calibration_records']} records), "
         f"{v['n_validation']} validation signatures ({v['n_validation_records']} records). Fitted rate R = "
         f"{v['R'] / 1e12:.3f} TFLOP/s, {100 * v['R'] / s['fp32_peak_flops']:.0f}% of the {s['fp32_peak_flops'] / 1e12:.1f} "
         "TFLOP/s FP32 datasheet peak.\n",
         "| metric | all | CONV | FC | all, seeds 0–99: median (5th–95th pct) |", "|---|---|---|---|---|"]
    var = res["validation_seed_variability"]
    for k, lab, f in (("abs_rel_median", "absolute relative error, median", pct),
                      ("abs_rel_p90", "absolute relative error, 90th percentile", pct),
                      ("abs_rel_p95", "absolute relative error, 95th percentile", pct),
                      ("abs_rel_max", "absolute relative error, maximum", pct),
                      ("signed_rel_median", "signed relative error, median", pct),
                      ("weighted_abs", "weighted absolute error Σ|pred−meas|/Σmeas", pct),
                      ("summed_error", "error of the summed time (Σpred−Σmeas)/Σmeas", pct),
                      ("spearman", "Spearman rank correlation (global)", lambda x: f"{x:.3f}")):
        L.append(f"| {lab} | {f(v['all'][k])} | {f(v['conv'].get(k))} | {f(v['fc'].get(k))} | "
                 f"{f(var[k]['median'])} ({f(var[k]['p5'])}–{f(var[k]['p95'])}) |")
    L += ["", f"The declared split (seed {v['seed']}) has a median absolute relative error at the "
          f"{var['abs_rel_median']['seed0_percentile']:.0f}th percentile of the 100 splits; the other seeds only show the "
          "split sensitivity.", "", f"Fraction of validation signatures overpredicted: {pct(v['all']['fraction_overpredicted'])}. "
          f"Spearman over all {c['signatures_measured']} signatures (split-independent): {res['spearman_all_signatures']:.3f}. "
          "Summed errors are over kernel-signature samples, not over any network's iteration.\n",
          "### Same calibration signatures, other model forms (comparison only; the planner uses the zero-intercept form)\n",
          "| model | estimator | intercept (ms) | rate (TFLOP/s) | median | p90 | p95 | weighted |", "|---|---|---|---|---|---|---|---|"]
    for m in res["model_comparison"]:
        L.append(f"| {m['model']} | {m['estimator']} | {m['intercept_ms']:.3f} | {m['rate_tflops']:.2f} | "
                 f"{pct(m['abs_rel_median'])} | {pct(m['abs_rel_p90'])} | {pct(m['abs_rel_p95'])} | {pct(m['weighted_abs'])} |")
    cr = res["configuration_ranking"]
    dd, cc, eq = cr["distinct_flop_pairs"], cr["consecutive_count_pairs"], cr["equal_flop_dp_mp_pairs"]
    L += ["", "## Configuration ranking within a layer (split-independent)\n",
          f"Over {cr['layers_with_two_or_more_configurations']} layers with at least two measured configurations, "
          "using the time the planner reads for each configuration:\n",
          f"* Pairs with different local FLOPs: {dd['pairs']}; measured order agrees with the FLOP order in "
          f"{pct(dd['fraction_concordant'])} (" + ", ".join(f"{w} {pct(x['fraction_concordant'])}"
                                                           for w, x in cr["by_workload"].items()) + ").",
          f"* Same strategy, consecutive measured counts: {cc['pairs']} pairs; {pct(cc['fraction_concordant'])} "
          "measured faster at the larger count.",
          f"* DP(p) and MP(p) pairs with equal local FLOPs (the constant-throughput model cannot order them): "
          f"{eq['n']}; measured slower/faster ratio median {eq['ratio_median']:.2f}, 90th percentile "
          f"{eq['ratio_p90']:.2f}; ratio above 1.25 in {pct(eq['fraction_ratio_above_1_25'])}; DP faster in "
          f"{pct(eq['fraction_dp_faster'])} ({eq['n_dp_faster']}), MP faster in {eq['n_mp_faster']}; "
          f"{eq['n_p1_same_kernel']} pairs are p = 1, where DP and MP time the same kernel (ratio 1). For the "
          f"{eq['p_above_1']['n']} pairs with p > 1: median ratio {eq['p_above_1']['ratio_median']:.2f}, 90th "
          f"percentile {eq['p_above_1']['ratio_p90']:.2f}.\n",
          "Global Spearman correlation measures ordering across shapes of very different size; the within-layer "
          "figures above measure the ordering among competing configurations of one layer.\n",
          "## Direct-FLOP-equivalent rate by shape size (one point per measured signature)\n",
          "| 3 × FLOPs per product | signatures | median (TFLOP/s) | 10th–90th percentile | share of measured time |",
          "|---|---|---|---|---|"]
    for b in res["rate_by_size"]:
        L.append(f"| {b['lo']:.0e}–{b['hi']:.0e} | {b['n']} | {b['median_tflops']:.2f} | "
                 f"{b['p10_tflops']:.2f}–{b['p90_tflops']:.2f} | {b['share_of_time']:.3f} |")
    ap = res["above_peak"]
    L += ["", f"{ap['signature_products']} signature products exceed the FP32 peak when their FLOPs are counted as "
          f"direct convolutions ({', '.join(f'{k}: {n}' for k, n in ap['by_shape'].items())}). {ap['note']}.\n",
          "## DP scaling efficiency by local batch\n",
          "Efficiency T(p0)·p0 / (T(p)·p), p0 = the layer's smallest measured DP count (1 = linear speedup). "
          "For the following layers p0 > 1, so their reference is not the full batch (the full-batch shape "
          "exceeded the notebook's memory cap): "
          + ", ".join(f"{x['workload']}/{x['layer']} (p0 = {x['p0']})" for x in res["scaling_efficiency"]["layers_with_p0_above_1"])
          + ".\n", "| local batch | layer configurations | median | 10th–90th percentile |", "|---|---|---|---|"]
    for b, x in res["scaling_efficiency"]["by_local_batch"].items():
        L.append(f"| {b} | {x['n']} | {x['median']:.2f} | {x['p10']:.2f}–{x['p90']:.2f} |")
    if res["reduction"]["median_rate_byte_s"]:
        L += ["", "## Local reduction\n",
              f"{res['reduction']['definition']}: {res['reduction']['median_rate_byte_s'] / 1e9:.1f} GB/s "
              f"({res['reduction']['n']} measurements). The rate at which the planner's reduction term reproduces the "
              "measured time; not a DRAM bandwidth measurement."]
    o = res["superseded_notebook_split"]
    L += ["", "## Superseded: the notebook's record-ID split (documents finding T1)\n",
          f"{o['n_calibration']} calibration and {o['n_validation']} validation records; {o['leaking_signatures']} "
          f"signatures on both sides, {o['validation_records_with_calibration_twin']} validation records with a "
          f"calibration twin. Recomputed from the CSV (agrees with the notebook's JSON: {o['agrees_with_notebook_json']}): "
          f"median {pct(o['validation']['abs_rel_median'])}, p90 {pct(o['validation']['abs_rel_p90'])}, p95 "
          f"{pct(o['validation']['abs_rel_p95'])}, maximum {pct(o['validation']['abs_rel_max'])}, signed median "
          f"{pct(o['validation']['signed_rel_median'])}, weighted {pct(o['validation']['weighted_abs'])}, summed "
          f"{pct(o['validation']['summed_error'])}, Spearman {o['validation']['spearman']:.3f}. Not used for any claim.",
          "", f"Cross-check of the reviewer's illustrative string-typed grouped split: median "
          f"{pct(res['cross_check_string_typed_signatures']['abs_rel_median'])} on "
          f"{res['cross_check_string_typed_signatures']['n_validation']} validation signatures."]
    out.write_text("\n".join(L) + "\n")


def plot(res: dict, stem=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8984"
    f = res["figure"]
    F, T, kind = np.array(f["all_F"]), np.array(f["all_T"]), np.array(f["all_kind"])
    vT, vp, vk = np.array(f["val_T"]), np.array(f["val_pred"]), np.array(f["val_kind"])
    R, peak = f["R"], res["source"]["fp32_peak_flops"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for k, col, lab in (("conv", BLUE, "CONV"), ("fc", ORANGE, "FC")):
        m = kind == k
        ax[0].scatter(F[m], F[m] / T[m] / 1e12, s=9, color=col, alpha=0.55, linewidths=0, label=lab)
        m = vk == k
        ax[1].scatter(vT[m] * 1e3, vp[m] * 1e3, s=9, color=col, alpha=0.55, linewidths=0, label=lab)
    ax[0].axhline(peak / 1e12, color=MUTED, lw=1, ls="--")
    ax[0].text(F.min() * 1.3, peak / 1e12 * 1.08, "FP32 peak (datasheet)", color=MUTED, fontsize=8)
    ax[0].axhline(R / 1e12, color=INK, lw=1)
    ax[0].text(F.min() * 1.3, R / 1e12 * 1.08, f"fitted R = {R / 1e12:.2f}", color=INK, fontsize=8)
    ax[0].set_xlabel("FLOPs of the kernel signature (3 products)")
    ax[0].set_ylabel("direct-FLOP-equivalent rate (TFLOP/s)")
    lo, hi = vT.min() * 1e3 * 0.7, vT.max() * 1e3 * 1.4
    ax[1].plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    ax[1].set_xlabel("measured time (ms), validation signatures")
    ax[1].set_ylabel("predicted time (ms)")
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
        a.legend(fontsize=8, frameon=False, loc="lower right")
        a.grid(alpha=0.25, which="major")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_title("(a) direct-FLOP-equivalent rate by signature", fontsize=9, loc="left")
    ax[1].set_title("(b) constant-throughput model, held-out signatures", fontsize=9, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160)
    plt.close(fig)


def main(zpath=ZIP):
    res = analyse(zpath)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    res = json.loads(OUT_JSON.read_text())          # report and figure from the stored analysis only
    write_markdown(res)
    plot(res)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ZIP)
