#!/usr/bin/env python
"""Convert a returned Colab profiling ZIP into a per-layer compute table for a
separately labeled `profiled_<gpu>` experiment (D17).

The table maps (workload, layer, strategy, p) -> the median forward,
input-gradient and weight-gradient times measured at that local shape; the
input-gradient time stays separate so the raw-input-gradient flag can drop it.

The ZIP is rejected (exit status 2, nothing written) unless
  * the notebook's manifest check reported a match;
  * the manifest hashes embedded in the notebook equal this repository's
    configs/workloads/SHA256SUMS for every profiled workload;
  * the profiled global batch equals --B (default: the nominal B);
  * precision was float32 with TF32 disabled for matmul and cuDNN;
  * every measured entry's local dimensions and attributes (kind, input shape,
    output features, local batch, kernel, stride, padding, bias) equal the local
    partition shape the planner evaluates for that (layer, strategy, p);
  * every measured kernel has at least MIN_REPS samples, and its median and quartile
    times are finite and positive with p25 <= median <= p75;
  * the summary CSV has every required column;
  * notebook v3 and later: no two measured records share a timed kernel signature, and
    a nonempty resume-fingerprint marker is present. The notebook enforces resume
    consistency; this ingester does not independently authenticate that history.
The parent layer's full output width is taken per use when the notebook records it
there (v3: one record per timed kernel signature, used by layers of different widths)
and from the row otherwise (v2).
Unmeasured shapes stay unmeasured and are listed; the planner refuses to use a
table that lacks a configuration it evaluates. Nothing is divided by p or
rescaled to another GPU.

Usage: python scripts/ingest_colab.py lhas_profile_<gpu>.zip [--B 1024] [--out configs/profiles]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lhas.model.workload import load_workload  # noqa: E402
from lhas.model.layercost import DP, MP  # noqa: E402
from lhas.model.profile import expected_local_shape  # noqa: E402

NOMINAL = json.loads((ROOT / "configs" / "nominal_experiment.json").read_text())
PARTS = ("forward", "input_grad", "weight_grad")
REQUIRED_COLUMNS = (["shape_id", "kind", "x", "n_out_full", "n_out_local", "batch_local", "kernel", "stride",
                     "padding", "bias", "uses", "status", "flops_per_product"]
                    + [f"{k}_{s}" for k in PARTS for s in ("median_s", "p25_s", "p75_s", "n")])
SIGNATURE_FIELDS = ("kind", "x", "n_out_local", "batch_local", "kernel", "stride", "padding", "bias")


class Rejected(Exception):
    pass


def repo_hashes() -> dict:
    out = {}
    for line in (ROOT / "configs" / "workloads" / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, f = line.split()
            out[Path(f).stem] = h
    return out


def _row_shape(r: dict, use: dict | None = None) -> dict:
    """Local shape of a record for one use. v3 records the parent width per use (a JSON
    list of widths in the row); v2 records one width per row."""
    if use is not None and "n_out_full" in use:
        n_full = int(use["n_out_full"])
    else:
        n_full = json.loads(r["n_out_full"])
        if isinstance(n_full, list):
            raise Rejected(f"shape_id {r['shape_id']}: several parent widths {n_full} but none recorded per use")
        n_full = int(n_full)
    return dict(kind=r["kind"], x=json.loads(r["x"]), n_out_full=n_full,
                n_out_local=int(r["n_out_local"]), batch_local=int(r["batch_local"]),
                kernel=json.loads(r["kernel"]), stride=json.loads(r["stride"]), padding=json.loads(r["padding"]),
                bias=r["bias"] in ("True", "true", "1", True))


def _timing_problems(r: dict) -> list:
    bad = []
    for k in PARTS:
        try:
            q1, md, q3 = (float(r[f"{k}_{s}"]) for s in ("p25_s", "median_s", "p75_s"))
        except (TypeError, ValueError):
            bad.append(f"{k}: non-numeric timing")
            continue
        if not all(math.isfinite(v) and v > 0 for v in (q1, md, q3)):
            bad.append(f"{k}: non-finite or non-positive timing")
        elif not q1 <= md <= q3:
            bad.append(f"{k}: quartiles out of order")
    return bad


def _signature(r: dict) -> tuple:
    return tuple(r[f] for f in SIGNATURE_FIELDS)


def ingest(zip_path: str, B: int) -> dict:
    z = zipfile.ZipFile(zip_path)
    names = {Path(n).name: n for n in z.namelist()}
    for req in ("environment.json", "config.json", "manifest_check.json", "layer_timings_summary.csv"):
        if req not in names:
            raise Rejected(f"{req} missing from the ZIP (notebook version 2 or later required)")
    env = json.loads(z.read(names["environment.json"]))
    cfg = json.loads(z.read(names["config.json"]))
    chk = json.loads(z.read(names["manifest_check.json"]))
    if chk.get("status") != "match":
        raise Rejected(f"notebook manifest check: {chk.get('status')}: {chk.get('problems', [])[:5]}")
    if cfg.get("B_GLOBAL") != B:
        raise Rejected(f"profiled for B={cfg.get('B_GLOBAL')}, requested B={B}")
    if env.get("precision") != "float32" or env.get("allow_tf32_matmul") or env.get("allow_tf32_cudnn"):
        raise Rejected(f"precision {env.get('precision')}, TF32 matmul {env.get('allow_tf32_matmul')}, "
                       f"TF32 cuDNN {env.get('allow_tf32_cudnn')} (float32 without TF32 required)")
    have = repo_hashes()
    emb = cfg.get("expected_manifest_sha256", {})
    bad = {w: (emb.get(w), have.get(w)) for w in emb if emb.get(w) != have.get(w)}
    if bad or not emb:
        raise Rejected(f"manifest hashes embedded in the notebook differ from this repository: {bad or 'none embedded'}")
    min_reps = int(cfg.get("MIN_REPS", 1))
    reader = csv.DictReader(io.TextIOWrapper(z.open(names["layer_timings_summary.csv"]), "utf-8"))
    absent = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if absent:
        raise Rejected(f"layer_timings_summary.csv lacks required columns {absent}")
    rows = list(reader)
    version = int(str(cfg.get("notebook_version", "0")).split(".")[0] or 0)
    if "resume_fingerprint.json" in names:
        fp = json.loads(z.read(names["resume_fingerprint.json"]))
        if not fp.get("fingerprint"):
            raise Rejected("resume_fingerprint.json has no fingerprint")
    elif version >= 3:
        raise Rejected("notebook v3 or later must include resume_fingerprint.json")
    # a resumed notebook run appends a new row for a shape that an earlier pass left
    # unmeasured (time budget); the last row of each shape_id is the valid one
    last = {}
    for r in rows:
        last[r["shape_id"]] = r
    rows = list(last.values())
    if version >= 3:
        seen = {}
        for r in rows:
            if r["status"] == "measured":
                sig = _signature(r)
                if sig in seen:
                    raise Rejected(f"shape_ids {seen[sig]} and {r['shape_id']} time the same kernel signature")
                seen[sig] = r["shape_id"]
    wls = {w: load_workload(w) for w in emb}
    table, missing, mismatches = {}, [], []
    for r in rows:
        uses = json.loads(r["uses"])
        for u in uses:
            key = f'{u["workload"]}|{u["layer"]}|{u["strategy"]}|{u["p"]}'
            if r["status"] != "measured":
                missing.append(dict(key=key, status=r["status"]))
                continue
            wl = wls[u["workload"]]
            L = wl.layer(u["layer"])
            c = (DP if u["strategy"] == "DP" else MP, int(u["p"]))
            exp = expected_local_shape(wl, L, c, B)
            got = _row_shape(r, u)
            diff = {f: (got[f], exp[f]) for f in got if got[f] != exp[f]}
            if diff:
                mismatches.append(dict(key=key, shape_id=int(r["shape_id"]), differences=diff))
                continue
            ns = [int(r[f"{k}_n"]) for k in PARTS]
            if min(ns) < min_reps:
                mismatches.append(dict(key=key, shape_id=int(r["shape_id"]), differences=dict(samples=ns)))
                continue
            tp = _timing_problems(r)
            if tp:
                mismatches.append(dict(key=key, shape_id=int(r["shape_id"]), differences=dict(timings=tp)))
                continue
            table[key] = dict(got, shape_id=int(r["shape_id"]),
                              **{k: float(r[f"{k}_median_s"]) for k in PARTS},
                              **{f"{k}_iqr_s": [float(r[f"{k}_p25_s"]), float(r[f"{k}_p75_s"])] for k in PARTS},
                              samples=min(ns))
    if mismatches:
        raise Rejected(f"{len(mismatches)} entries do not match the planner's local shapes, e.g. {mismatches[:3]}")
    label = env["input_mode_label"]
    return dict(input_mode=label, B=B, device=env["gpu_name"], precision="float32", tf32=False,
                manifest_sha256={w: have[w] for w in emb}, notebook_version=cfg.get("notebook_version"),
                source_zip_sha256=hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
                environment=env, config=cfg, table=table, unmeasured=missing)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--B", type=int, default=NOMINAL["B"])
    ap.add_argument("--out", default=str(ROOT / "configs" / "profiles"))
    a = ap.parse_args(argv)
    try:
        t = ingest(a.zip, a.B)
    except Rejected as e:
        print("REJECTED:", e, file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{t['input_mode']}_B{t['B']}.json"
    f.write_text(json.dumps(t, indent=1))
    print(f"{t['input_mode']}: {len(t['table'])} measured layer/configuration entries, "
          f"{len(t['unmeasured'])} unmeasured -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
