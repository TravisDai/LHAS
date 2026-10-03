"""Export a constructed operation as a plain schedule record for the independent
verifier (`lhas_verify`), plus the semantic specification of the collective.

The record lists every route with its explicit hop node sequence, labels and
the constructor's explicit chunk transmissions (hop, chunk, step, bytes), and
per-round claimed setup flags, step counts and step-duration sums as computed
by the numba timing kernel. The verifier recomputes all of these.
"""
from __future__ import annotations

import numpy as np

from ..params import TransportParams
from .schedule import Operation, evaluate, hop_nodes


def params_dict(tp: TransportParams, beta_red: float) -> dict:
    return dict(N=tp.N, lam=tp.lam, w_tx=tp.w_tx, w_rx=tp.w_rx, reach=tp.reach, d_peer=tp.d_peer,
                kappa_bytes=tp.kappa_bytes, rate_bps=tp.rate_bps, eps=tp.eps, t_tx=tp.t_tx,
                t_rx=tp.t_rx, t_fwd=tp.t_fwd, seg_len_m=tp.seg_len_m, v_g=tp.v_g,
                t_setup=tp.t_setup, beta_red=beta_red)


def _chunk_sizes(V: int, kappa: int) -> list[int]:
    if kappa <= 0:
        return [V]
    c = -(-V // kappa)
    return [min(kappa, V - (j - 1) * kappa) for j in range(1, c + 1)]


def export_record(op: Operation, payload, tp: TransportParams, beta_red: float,
                  include_tx: bool = True) -> dict:
    ev = evaluate(op, payload, tp, beta_red, detail=True)
    L, N = tp.eff_reach, tp.N
    p_arr = np.asarray(payload, np.int64)
    phases = []
    for ph, (setup, st, body) in zip(op.phases, ev.per_round):
        rec = dict(name=ph.name, semantics=ph.semantics, rule=ph.rule, rounds=[])
        rec.update(ph.rule_meta)
        if ph.reduce_counts is not None:
            rec["reduce_counts"] = {int(i + 1): int(a) for i, a in enumerate(ph.reduce_counts) if a > 0}
        for q in range(ph.nrounds):
            idx = ph.order_by_round[ph.round_start[q]:ph.round_start[q + 1]]
            pipelined = bool(ph.pipelined[q])
            routes = []
            for r in idx:
                r = int(r)
                V = int(p_arr) if p_arr.ndim == 0 else int(p_arr[ph.pay[r]])
                nodes = hop_nodes(int(ph.src[r]), int(ph.dirn[r]), int(ph.H[r]), int(ph.last[r]), L, N)
                labs = ph.labels[ph.hop_start[r]:ph.hop_start[r + 1]]
                hops = []
                for k in range(int(ph.H[r])):
                    ns = L if k < ph.H[r] - 1 else int(ph.last[r])
                    hops.append([nodes[k], nodes[k + 1], int(ph.dirn[r]), ns, int(labs[k])])
                route = dict(src=int(ph.src[r]), dst=int(ph.dst[r]), item=int(ph.item[r]), bytes=V, hops=hops)
                if include_tx:
                    sizes = _chunk_sizes(V, tp.kappa_bytes) if pipelined else [V]
                    route["tx"] = [[a, j, a + j - 1, sz] for a in range(1, len(hops) + 1)
                                   for j, sz in enumerate(sizes, start=1)]
                routes.append(route)
            rec["rounds"].append(dict(routes=routes, setup=int(setup[q]), steps=int(st[q]), body=float(body[q])))
        phases.append(rec)
    nz = lambda a: {int(i + 1): float(x) for i, x in enumerate(a) if x > 0}
    return dict(op=op.op, candidate=op.candidate, phases=phases,
                claimed=dict(transport=ev.transport, reduction=ev.reduction, steps=ev.steps,
                             setups=ev.setups, rounds=ev.rounds, relay_peak=nz(ev.relay_peak),
                             gathered=nz(ev.gathered), recv_peak=nz(ev.recv_buffers),
                             accumulators=nz(ev.accumulators)))


# ----------------------------------------------------------------------------
# semantic specifications
# ----------------------------------------------------------------------------
def spec_allgather(P, Q, shard_bytes: int, initial=None) -> dict:
    init = {u: [u] for u in P} if initial is None else {u: sorted(v) for u, v in initial.items()}
    return dict(P=list(P), Q=list(Q), initial=init, required={v: list(P) for v in Q},
                shard_bytes={u: int(shard_bytes) for u in P})


def spec_allreduce(participants, contribution_bytes: int) -> dict:
    return dict(participants=list(participants), contribution_bytes=int(contribution_bytes))


def spec_alltoall(src, dst, nbytes) -> dict:
    req = {}
    for u, v, b in zip(src, dst, nbytes):
        if int(u) != int(v) and int(b) > 0:
            req[(int(u), int(v))] = req.get((int(u), int(v)), 0) + int(b)
    return dict(required=req)
