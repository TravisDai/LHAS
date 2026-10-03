#!/usr/bin/env python
"""CPU-only smoke test of the Colab notebook logic and of the ingestion safeguards.

No GPU timing is performed and nothing produced here is a measurement: CUDA-event
timing is replaced by wall-clock timing of a handful of tiny CPU shapes so that the
code paths (manifest comparison, incremental CSV writing and resume, try/finally
cleanup, reduction benchmark, held-out validation statistics, ZIP packaging and
ingestion checks) execute. Outputs go to a temporary directory only.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import nbformat
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _cells():
    nb = nbformat.read(str(HERE / "LHAS_compute_profiling.ipynb"), as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def run(tmp: Path, verbose: bool = True) -> dict:
    cells = _cells()
    out_dir = tmp / "lhas_profile_cpu-smoke"
    props = types.SimpleNamespace(name="cpu-smoke", total_memory=8 * 2 ** 30, multi_processor_count=0, major=0, minor=0)
    loop = cells[5]
    reductions = cells[6].replace("[1, 4, 16, 64, 256]", "[1]").replace("[1, 2, 4, 8]", "[1, 2]")
    small_keys = []

    def session(stamp: str, cfg_edit=None) -> dict:
        """One notebook session (Runtime > Run all) up to, not including, the profiling cell;
        cfg_edit changes the configuration cell (a different configuration on resume)."""
        g = {"__name__": "nb"}
        env = {"gpu_name": "cpu-smoke", "precision": "float32", "allow_tf32_matmul": False, "allow_tf32_cudnn": False,
               "input_mode_label": "profiled_cpu-smoke", "torch": torch.__version__, "timestamp_utc": stamp,
               "note": "CPU smoke test, not a measurement"}
        exec("import json, os, platform, subprocess, sys, time, datetime, math, random, csv, zipfile, hashlib\n"
             "import torch, torchvision\n", g)
        g.update(dev=torch.device("cpu"), props=props, ENV=env)
        c1 = cells[1].replace('OUT = f"/content/lhas_profile_', f'OUT = f"{tmp}/lhas_profile_')
        exec(cfg_edit(c1) if cfg_edit else c1, g)                         # config
        exec(cells[2], g)                                                 # export + manifest comparison
        assert g["CHECK"]["status"] == "match", g["PROBLEMS"][:5]
        exec(cells[3], g)                                                 # local shapes
        exec(cells[4], g)                                                 # helpers

        def cpu_timed(fn, warmup=1, reps=g["MIN_REPS"]):
            fn()
            s = []
            for _ in range(reps):
                t = time.perf_counter(); fn(); s.append(time.perf_counter() - t)
            return s
        g["timed"] = cpu_timed
        # restrict to the smallest shapes so that the CPU run is short (same shapes every session)
        if not small_keys:
            small_keys.extend(k for k, _ in sorted(g["SHAPES"].items(), key=lambda kv: g["est_bytes"](kv[0]))[:24])
        g["n_shapes_all"] = len(g["SHAPES"])
        g["SHAPES"] = {k: g["SHAPES"][k] for k in small_keys}
        return g

    files = ("environment.json", "config.json", "manifest_check.json", "shape_manifest.json",
             "layer_timings_summary.csv", "layer_timings_raw.csv", "reduction_timings.csv", "resume_fingerprint.json")
    snap = lambda: {f: (out_dir / f).read_bytes() if (out_dir / f).exists() else None for f in files}  # noqa: E731
    g = session("2026-01-01T00:00:00+00:00")
    n_shapes = g["n_shapes_all"]
    small = small_keys
    exec(loop, g)                                                         # profile (first pass)
    exec(reductions, g)
    first = list(csv.DictReader(open(out_dir / "layer_timings_summary.csv")))
    before = snap()
    g = session("2026-01-02T00:00:00+00:00")                              # compatible resume (new session)
    exec(loop, g)
    exec(reductions, g)
    second = list(csv.DictReader(open(out_dir / "layer_timings_summary.csv")))
    assert len(first) == len(second) == len(small)
    after = snap()
    compatible_resume_unchanged = all(before[f] == after[f] for f in files if f != "resume_fingerprint.json")
    fp = json.loads((out_dir / "resume_fingerprint.json").read_text())
    assert len(fp["resumed_utc"]) == 1
    # a resume under a different configuration must be refused and must modify nothing
    try:
        g2 = session("2026-01-03T00:00:00+00:00", cfg_edit=lambda c: c.replace("REPS = 30 ", "REPS = 31 "))
        assert g2["REPS"] == 31
        exec(loop, g2)
        resume_guard = False
    except RuntimeError:
        resume_guard = True
    rejected_resume_unchanged = snap() == after
    assert resume_guard and len(list(csv.DictReader(open(out_dir / "layer_timings_summary.csv")))) == len(small)
    # v3: one record per timed kernel signature (no parent width in the key)
    sig = lambda r: tuple(r[f] for f in ("kind", "x", "n_out_local", "batch_local", "kernel", "stride", "padding", "bias"))
    assert len({sig(r) for r in second}) == len(second)
    exec(cells[7], g)                                                     # validation
    assert g["VALIDATION"]["status"] == "insufficient samples"
    # validation statistics on synthetic rows: a perfect constant-throughput device
    rows = [dict(shape_id=i, status="measured", kind="conv" if i % 2 else "fc", flops_per_product=1e9 * (i + 1),
                 **{f"{k}_median_s": 1e-3 * (i + 1) for k in g["PARTS"]}) for i in range(60)]
    v = g["validate"](rows)
    assert v["status"] == "ok" and abs(v["validation"]["absolute_relative_error_median"]) < 1e-9
    assert abs(v["validation"]["spearman_rank_correlation"] - 1.0) < 1e-12
    # kernel decomposition agrees with autograd
    torch.manual_seed(0)
    ks, flops, (x, w, b, y, gy) = g["conv_kernels"]((3, 9, 9), 4, (3, 3), (2, 2), (1, 1), True, 2)
    x.requires_grad_(); w.requires_grad_()
    gx, gw = torch.autograd.grad(torch.nn.functional.conv2d(x, w, None, (2, 2), (1, 1)), (x, w), gy)
    assert torch.allclose(ks["input_grad"]()[0], gx, atol=1e-4) and torch.allclose(ks["weight_grad"]()[1], gw, atol=1e-4)
    ks, flops, (x, w, b, gy) = g["fc_kernels"](7, 5, False, 3)
    x.requires_grad_(); w.requires_grad_()
    gx, gw = torch.autograd.grad(torch.nn.functional.linear(x, w), (x, w), gy)
    assert torch.allclose(ks["input_grad"](), gx, atol=1e-5) and torch.allclose(ks["weight_grad"](), gw, atol=1e-5)
    # package and ingest
    zip_path = shutil.make_archive(str(out_dir), "zip", str(out_dir))
    import ingest_colab as IC
    t = IC.ingest(zip_path, 1024)
    measured = sum(r["status"] == "measured" for r in second)
    uses = sum(len(json.loads(r["uses"])) for r in second if r["status"] == "measured")
    assert len(t["table"]) == uses and measured > 0
    # rejection paths
    rejected = {}

    def tamper(name, fn):
        d = tmp / f"tamper_{name}"
        shutil.copytree(out_dir, d)
        fn(d)
        z = shutil.make_archive(str(d), "zip", str(d))
        try:
            IC.ingest(z, 1024)
            rejected[name] = False
        except IC.Rejected:
            rejected[name] = True

    def edit_json(fname, key, val):
        def f(d):
            p = d / fname
            j = json.loads(p.read_text()); j[key] = val; p.write_text(json.dumps(j))
        return f

    def edit_rows(d):
        p = d / "layer_timings_summary.csv"
        rows = list(csv.DictReader(open(p)))
        i = next(k for k, r in enumerate(rows) if r["status"] == "measured")
        rows[i]["batch_local"] = str(int(rows[i]["batch_local"]) + 1)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # resumed run: an earlier "time budget" row for a shape followed by its measured row
    d = tmp / "resumed"
    shutil.copytree(out_dir, d)
    pth = d / "layer_timings_summary.csv"
    rows_r = list(csv.DictReader(open(pth)))
    first = next(r for r in rows_r if r["status"] == "measured")
    stale = {k: ("" if k.endswith(("_s", "_n")) or k == "flops_per_product" else v) for k, v in first.items()}
    stale["status"] = "not measured: time budget exhausted"
    with open(pth, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_r[0])); w.writeheader(); w.writerows([stale] + rows_r)
    t2 = IC.ingest(shutil.make_archive(str(d), "zip", str(d)), 1024)
    keys = {f'{u["workload"]}|{u["layer"]}|{u["strategy"]}|{u["p"]}' for u in json.loads(first["uses"])}
    assert keys <= set(t2["table"]) and not keys & {m["key"] for m in t2["unmeasured"]}
    tamper("batch", edit_json("config.json", "B_GLOBAL", 256))
    tamper("manifest_hash", edit_json("config.json", "expected_manifest_sha256", {"alexnet": "0" * 64}))
    tamper("manifest_check", edit_json("manifest_check.json", "status", "mismatch"))
    tamper("tf32", edit_json("environment.json", "allow_tf32_matmul", True))
    tamper("local_dims", edit_rows)

    def nan_timing(d):
        p = d / "layer_timings_summary.csv"
        rows = list(csv.DictReader(open(p)))
        i = next(k for k, r in enumerate(rows) if r["status"] == "measured")
        rows[i]["forward_median_s"] = "nan"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def drop_column(d):
        p = d / "layer_timings_summary.csv"
        rows = list(csv.DictReader(open(p)))
        keep = [c for c in rows[0] if c != "weight_grad_p75_s"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    def duplicate_signature(d):
        p = d / "layer_timings_summary.csv"
        rows = list(csv.DictReader(open(p)))
        r = dict(next(r for r in rows if r["status"] == "measured"))
        r["shape_id"] = str(max(int(x["shape_id"]) for x in rows) + 1)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows + [r])

    tamper("nan_timing", nan_timing)
    tamper("missing_column", drop_column)
    tamper("duplicate_signature", duplicate_signature)
    tamper("no_fingerprint", lambda d: (d / "resume_fingerprint.json").unlink())
    assert all(rejected.values()), rejected
    res = dict(shapes_total=n_shapes, shapes_run=len(small), measured=measured, table_entries=len(t["table"]),
               rejections=rejected, resume_guard=resume_guard, rejected_resume_unchanged=rejected_resume_unchanged,
               compatible_resume_unchanged=compatible_resume_unchanged)
    if verbose:
        print(json.dumps(res, indent=1))
        print("CPU smoke test passed (no GPU timing performed; outputs are not measurements)")
    return res


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        run(Path(d))
