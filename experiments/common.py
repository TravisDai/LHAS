"""Shared helpers for experiment runners: nominal specification, argument parsing,
parameter construction, provenance and JSON serialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lhas.params import load_approved  # noqa: E402

NOMINAL = json.loads((ROOT / "configs" / "nominal_experiment.json").read_text())


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--workload", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--B", type=int, default=NOMINAL["B"])
    ap.add_argument("--raw", choices=["retain", "omit"], default=NOMINAL["raw_input_grad"])
    ap.add_argument("--norm", choices=["sync", "local", "fixed", "none"], default=NOMINAL["norm_policy"])
    ap.add_argument("--opt-state", type=float, default=NOMINAL["optimizer_state_factor"])
    ap.add_argument("--reserve", type=int, default=NOMINAL["runtime_reserve_bytes"],
                    help="runtime reserve h per node, inside the 12 GiB budget (bytes)")
    ap.add_argument("--workspace", type=int, default=NOMINAL["workspace_bytes"])
    ap.add_argument("--alpha", type=float, default=NOMINAL["alpha"])
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--red-eff", type=float, default=None)
    ap.add_argument("--lam", type=int, default=None)
    ap.add_argument("--w", type=int, default=None, help="Tx/Rx per node (default: approved)")
    ap.add_argument("--reach", type=int, default=None)
    ap.add_argument("--dpeer", type=int, default=None)
    ap.add_argument("--t-setup", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--kappa", type=int, default=None)
    ap.add_argument("--family", choices=["all", "one-stage", "deepest"], default="all")
    ap.add_argument("--baselines", default="single,dp,dpbest,owt,mcmc")
    ap.add_argument("--mcmc-iters", type=int, default=NOMINAL["mcmc"]["iterations"])
    ap.add_argument("--mcmc-seeds", type=int, default=len(NOMINAL["mcmc"]["seeds"]))
    ap.add_argument("--profile", default=None,
                    help="profiled compute table (scripts/ingest_colab.py output); replaces analytic T_cp in a "
                         "separately labeled experiment; unmeasured configurations are excluded")
    ap.add_argument("--red-rate", type=float, default=None,
                    help="local reduction rate beta_red in byte/s (overrides --red-eff)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out", default=str(ROOT / "results" / "raw"))
    return ap


def build_params(a):
    tp, cp = load_approved(a.N)
    tp = replace(tp, **{k: v for k, v in dict(lam=a.lam, reach=a.reach, d_peer=a.dpeer, t_setup=a.t_setup,
                                                  eps=a.eps, kappa_bytes=a.kappa).items() if v is not None})
    if a.w is not None:
        tp = replace(tp, w_tx=a.w, w_rx=a.w)
    cp = replace(cp, runtime_reserve_bytes=int(a.reserve),
                 **{k: v for k, v in dict(utilization=a.rho, reduction_efficiency=a.red_eff).items() if v is not None})
    if getattr(a, "red_rate", None):
        cp = replace(cp, reduction_efficiency=a.red_rate / cp.peak_mem_bw)
    if getattr(a, "profile", None):
        cp = replace(cp, input_mode=load_profile(a.profile)["input_mode"])
    tp.validate(); cp.validate()
    return tp, cp


def load_profile(path):
    return json.loads(Path(path).read_text())


def profile_provenance(path) -> dict:
    t = load_profile(path)
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                input_mode=t["input_mode"], device=t["device"], B=t["B"], source_zip_sha256=t["source_zip_sha256"],
                measured_entries=len(t["table"]), unmeasured_entries=len(t["unmeasured"]))


def source_digest() -> str:
    """sha256 over the (path, content) of every Python source under src/lhas and experiments,
    independent of git, so an output can be matched to a source tree."""
    h = hashlib.sha256()
    for f in sorted([*(ROOT / "src" / "lhas").rglob("*.py"), *(ROOT / "experiments").glob("*.py")]):
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def provenance(workload: str) -> dict:
    man = ROOT / "configs" / "workloads" / f"{workload}.json"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "src", "experiments"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        head, dirty = "unknown", True
    cache_dir = os.environ.get("LHAS_PACK_CACHE_DIR")
    pre = len(list(Path(cache_dir).glob("*.npz"))) if cache_dir and Path(cache_dir).exists() else 0
    diff_sha = None
    if dirty:          # identify the uncommitted source state that produced this output
        try:
            diff = subprocess.run(["git", "diff", "HEAD", "--", "src", "experiments"], cwd=ROOT,
                                  capture_output=True).stdout
            diff_sha = hashlib.sha256(diff).hexdigest()
        except Exception:
            pass
    nominal = ROOT / "configs" / "nominal_experiment.json"
    return dict(manifest=str(man.relative_to(ROOT)), manifest_sha256=hashlib.sha256(man.read_bytes()).hexdigest(),
                code_commit=head, code_dirty=dirty, uncommitted_diff_sha256=diff_sha,
                source_sha256=source_digest(), nominal_spec_sha256=hashlib.sha256(nominal.read_bytes()).hexdigest(),
                python=sys.version.split()[0],
                pack_cache_dir=cache_dir, pack_cache_files_at_start=pre,
                nominal_spec=str((ROOT / "configs" / "nominal_experiment.json").relative_to(ROOT)))


def cfg_str(c):
    return ("DP" if c[0] == 0 else "MP") + str(c[1])


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return x


def write(a, out: dict):
    d = Path(a.out) / a.tag
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{a.workload}_N{a.N}.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(jsonable(out), indent=1))
    tmp.replace(f)
    return f


def rss_peak():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
