"""Re-verify every selected operation of a plan with the independent verifier and
report coverage for each one.

Every network operation of the plan receives exactly one record:
  checked-actual   verified at its actual payload
  checked-reduced  structure verified at a reduced payload of 2*kappa+1 bytes per item
                   (three chunks with a short tail) because the explicit hop-chunk
                   transmission count at the actual payload exceeds MAX_TRANSMISSIONS
  skipped          not verified; the reason is recorded (route count above
                   MAX_VERIFY_ROUTES, the pure-Python verifier's resource limit)
and `packing_checked` states whether the deterministic first-fit was re-derived by
the pure-Python reference (routes <= MAX_PACKING_ROUTES). Local operations (one
participant) have no network schedule and are listed as `local`.
A failed check raises VerificationError (the run is marked failed, never skipped).
"""
from __future__ import annotations

import numpy as np

from lhas_verify import verify
from .collectives import families as F
from .model import layout as Lt
from .model.chain import ChainModel, DP, MP
from .transport.export import export_record, params_dict, spec_allgather, spec_allreduce, spec_alltoall

MAX_TRANSMISSIONS = 1_500_000
MAX_VERIFY_ROUTES = 150_000
MAX_PACKING_ROUTES = 20_000


def _find(ops_iter, cand):
    for op in ops_iter:
        if op.candidate == cand:
            return op
    raise KeyError(cand)


def check_operation(kind, op, payload, spec_fn, tp, beta, where=""):
    nroutes = sum(ph.nroutes for ph in op.phases)
    rec = dict(kind=kind, where=where, candidate=op.candidate, routes=int(nroutes))
    if nroutes > MAX_VERIFY_ROUTES:
        rec.update(status="skipped", reason=f"{nroutes} routes > MAX_VERIFY_ROUTES={MAX_VERIFY_ROUTES}",
                   packing_checked=False)
        return rec
    hops = sum(int(ph.H.sum()) for ph in op.phases)
    p_arr = np.asarray(payload)
    vmax = int(p_arr.max()) if p_arr.size else 0
    chunks = -(-vmax // tp.kappa_bytes) if tp.kappa_bytes > 0 else 1
    reduced = hops * chunks > MAX_TRANSMISSIONS
    if reduced:
        red = 2 * tp.kappa_bytes + 1 if tp.kappa_bytes > 0 else 1
        payload = np.minimum(p_arr, red) if p_arr.ndim else red
    packing = nroutes <= MAX_PACKING_ROUTES
    r = export_record(op, payload, tp, beta)
    out = verify(r, spec_fn(payload), params_dict(tp, beta), check_packing=packing)
    rec.update(status="checked-reduced" if reduced else "checked-actual", packing_checked=bool(packing),
               verified_transport=out["transport"], verified_reduction=out["reduction"])
    if reduced:
        rec["reason"] = f"{hops * chunks} hop-chunk transmissions > MAX_TRANSMISSIONS={MAX_TRANSMISSIONS}"
    return rec


def _local(kind, where):
    return dict(kind=kind, where=where, candidate="local", routes=0, status="local", packing_checked=False)


def _ar(kind, p, V, cand, tp, beta, where):
    if p <= 1 or cand in ("local", "empty"):
        return _local(kind, where)
    op = _find(F.iter_wrht_ops(list(range(1, p + 1)), tp), cand)
    return check_operation(kind, op, V, lambda v: spec_allreduce(range(1, p + 1), int(np.asarray(v).max())), tp, beta,
                           where)


def _ag(kind, p, q, V, cand, tp, beta, where):
    if (p <= 1 and q <= 1) or cand in ("local", "empty"):
        return _local(kind, where)
    op = _find(F.iter_allgather_ops(p, q, tp), cand)
    return check_operation(kind, op, V, lambda v: spec_allgather(range(1, p + 1), range(1, q + 1), int(np.asarray(v).max())),
                           tp, beta, where)


def _a2a(kind, s, d, b, tp, beta, where):
    if s.size == 0:
        return _local(kind, where)
    op = F.osm(s, d, tp)
    return check_operation(kind, op, np.asarray(b), lambda v: spec_alltoall(s, d, v), tp, beta, where)


def summarize(records):
    cnt = {}
    for r in records:
        cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    net = [r for r in records if r["status"] != "local"]
    return dict(expected_operations=len(records), network_operations=len(net), by_status=cnt,
                packing_checked=sum(1 for r in net if r["packing_checked"]),
                all_network_operations_checked=all(r["status"].startswith("checked") for r in net))


def verify_chain_plan(model: ChainModel, result) -> dict:
    tp, beta, psi, B = model.tp, model.cp.beta_red, model.psi, model.st.B
    recs = []
    Lc = len(model.L)
    for i in range(Lc):
        ci = result.configs[i]
        cj = result.configs[i + 1] if i + 1 < Lc else None
        choice = dict(result.choices[i].choices)
        L = model.L[i]
        t, p = ci
        where = L.name
        if "AR_w" in choice:
            recs.append(_ar("AR_w", p, psi * model.param_numel(L), choice["AR_w"], tp, beta, where))
        if "AR_x" in choice:
            recs.append(_ar("AR_x", p, psi * L.x_numel * B, choice["AR_x"], tp, beta, where))
        if "AG_y" in choice:
            q = cj[1] if cj is not None else p
            S = psi * (L.u_numel if cj is not None else L.y_numel) * B
            recs.append(_ag("AG_y", p, q, S // p, choice["AG_y"], tp, beta, where))
        if cj is None:
            continue
        for name, spec in model._inter_specs(i, ci, cj):
            s, d, b = spec[1:]
            recs.append(_a2a(name, s, d, b, tp, beta, f"{where}->{model.L[i + 1].name}"))
    return dict(summary=summarize(recs), operations=recs)


def verify_graph_plan(model, ops) -> dict:
    """ops: list of (where, OpChoice) from evaluate_graph_plan_full."""
    tp, beta = model.tp, model.cp.beta_red
    recs = []
    for where, o in ops:
        sp = o.spec
        if sp is None:
            recs.append(dict(kind=o.name, where=where, candidate=o.alt.cand, routes=0, status="skipped",
                             reason="no rebuild specification", packing_checked=False))
        elif sp[0] == "AR":
            recs.append(_ar(o.name, sp[1], sp[2], o.alt.cand, tp, beta, where))
        elif sp[0] == "AG":
            recs.append(_ag(o.name, sp[1], sp[2], sp[3], o.alt.cand, tp, beta, where))
        else:
            recs.append(_a2a(o.name, sp[1], sp[2], sp[3], tp, beta, where))
    return dict(summary=summarize(recs), operations=recs)


# backwards-compatible name
def verify_plan(model, result, **_):
    return verify_chain_plan(model, result)["operations"]


__all__ = ["verify_chain_plan", "verify_graph_plan", "check_operation", "summarize", "MAX_TRANSMISSIONS",
           "MAX_VERIFY_ROUTES", "MAX_PACKING_ROUTES"]
