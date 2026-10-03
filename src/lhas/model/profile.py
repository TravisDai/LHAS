"""Profiled compute tables (scripts/ingest_colab.py) and their safeguards (D17).

A table is valid for a run only if it was measured for the same global batch B,
the same workload manifests (sha256), float32 without TF32, and every entry's
local dimensions equal the local partition shape the planner evaluates:
  DP(p): local batch B/p, all n_out output features;
  MP(p): complete batch B, n_out/p output features.
Mismatches raise ProfileMismatch; missing entries raise KeyError (unmeasured
configurations are never extrapolated or rescaled)."""
from __future__ import annotations

from .layercost import DP


class ProfileMismatch(ValueError):
    pass


def layer_attributes(wl, L) -> dict:
    """Shape attributes of a weighted layer from its manifest node."""
    node = next((n for n in wl.manifest.get("nodes", []) if n["id"] == L.name), {})
    return dict(kind=L.kind, x=list(L.x_shape), n_out_full=L.n_out,
                kernel=list(node.get("kernel", [])) if L.kind == "conv" else [],
                stride=list(node.get("stride", [])) if L.kind == "conv" else [],
                padding=list(node.get("padding", [])) if L.kind == "conv" else [],
                bias=bool(node.get("bias", L.bias_numel > 0)))


def expected_local_shape(wl, L, c, B: int) -> dict:
    t, p = c
    d = layer_attributes(wl, L)
    if t == DP:
        d.update(batch_local=B // p, n_out_local=L.n_out)
    else:
        d.update(batch_local=B, n_out_local=L.n_out // p)
    return d


SHAPE_FIELDS = ("kind", "x", "n_out_full", "n_out_local", "batch_local", "kernel", "stride", "padding", "bias")


def check_table_header(table: dict, wl, B: int):
    if table.get("B") != B:
        raise ProfileMismatch(f"{table.get('input_mode')}: measured for B={table.get('B')}, run uses B={B}")
    want = wl.manifest_sha256
    got = (table.get("manifest_sha256") or {}).get(wl.name)
    if not want or got != want:
        raise ProfileMismatch(f"{table.get('input_mode')}: manifest hash for {wl.name} is {got}, repository has {want}")
    if table.get("precision") != "float32" or table.get("tf32", True):
        raise ProfileMismatch(f"{table.get('input_mode')}: precision {table.get('precision')}, "
                              f"TF32 {table.get('tf32')} (float32 without TF32 required)")


def profile_key(wl, L, c) -> str:
    return f"{wl.name}|{L.name}|{'DP' if c[0] == DP else 'MP'}|{c[1]}"


def has_entry(table: dict, wl, L, c) -> bool:
    return profile_key(wl, L, c) in table["table"]


def profiled_time(table: dict, wl, L, c, B: int, input_grad: bool) -> float:
    check_table_header(table, wl, B)
    key = f"{wl.name}|{L.name}|{'DP' if c[0] == DP else 'MP'}|{c[1]}"
    e = table["table"].get(key)
    if e is None:
        raise KeyError(f"no measured time for {key} in {table['input_mode']}; this configuration is unmeasured")
    exp = expected_local_shape(wl, L, c, B)
    bad = {f: (e.get(f), exp[f]) for f in SHAPE_FIELDS if f in e and e.get(f) != exp[f]}
    missing = [f for f in ("batch_local", "n_out_local") if f not in e]
    if bad or missing:
        raise ProfileMismatch(f"{key}: measured local shape differs from the planner's {bad or missing}")
    return e["forward"] + e["weight_grad"] + (e["input_grad"] if input_grad else 0.0)


__all__ = ["ProfileMismatch", "expected_local_shape", "layer_attributes", "check_table_header", "profiled_time", "has_entry", "profile_key",
           "SHAPE_FIELDS"]
