"""Evaluate a complete configuration assignment on a typed-interface graph:
additive cost, the reservation vector (Eq. memory_feasibility) and the list of
selected operations. Used by plan_graph (consistency with the search) and by
fixed-configuration baselines."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..model.branch import GraphModel, OpChoice
from ..model.chain import DP, MP
from ..model import layercost as _lc


def evaluate_graph_plan(model: GraphModel, cfg: dict):
    r = evaluate_graph_plan_full(model, cfg)
    return r["cost"], r["memory"], r["ops"]


def evaluate_graph_plan_full(model: GraphModel, cfg: dict) -> dict:
    """Cost, reservation vector with the minimum-charge alternative of every
    operation, a reservation lower bound over all retained alternatives, and a
    status: feasible; infeasible (even the lower bound exceeds the budget); or
    not_established (the fastest combination fails but slower ones are not excluded)."""
    gs = model.gs
    N, psi, B = model.N, model.psi, model.B
    cost = 0.0
    mem = np.zeros(N)
    ops = []
    for n in model.order:
        mem += model.m(n, cfg[n])
    # layer charges (with output All-gather for MP producers)
    for n in model.order:
        c = cfg[n]
        oi = gs.iface_of_output.get(n)
        if c[0] == MP:
            if oi is None:
                L = model.L[n]
                mask = np.ones(N); mask[:c[1]] = 0.0
                g, ch = model.charge(n, c, (c[1], c[1], (psi * L.y_numel * B) // c[1], mask))
            else:
                I = gs.interfaces[oi]
                prod = next(p for p in I.producers if p.kind == "layer" and p.ref == n)
                ccfg = {x: cfg[x] for x in I.consumers}
                q = max(x[1] for x in ccfg.values())
                mask = model.ag_mask(I, prod, ccfg, c[1])
                g, ch = model.charge(n, c, (c[1], q, (psi * model.rows(I, prod) * B) // c[1], mask))
                # keep every field (retained alternatives, verification specification); only the
                # All-gather mask changes (F1 of LHAS_0892134_Review)
                ch = [replace(o, mask=mask if o.name == "AG_y" else o.mask) for o in ch]
        else:
            g, ch = model.charge(n, c)
        cost += g
        ops += [(n, o) for o in ch]
    # interfaces
    for I in gs.interfaces:
        if I.join == "input":
            continue
        cons = I.consumers
        K = len(cons)
        for p in I.producers:
            if p.kind == "layer":
                cb = cfg[p.ref]
                for cn in cons:
                    cc = cfg[cn]
                    if cb[0] == DP:
                        t, ch = model.fp_pair_dp(I, p, cb, cc)
                        cost += t; ops += [(f"T{I.idx}", o) for o in ch]
                    t, ch = model.bp_pair(I, p, cb, cc)
                    cost += t; ops += [(f"T{I.idx}", o) for o in ch]
                V_loc = model.local_bytes_layer(I, p, cb)
                cost += model.local_sum_time(V_loc, K)
                if K >= 2:
                    mem[:cb[1]] += (K + 1) * V_loc          # B1: retained contributions + accumulator
                if not _lc.aliasable_ops(p.ops) or I.join == "add":
                    # U1: separately stored slice/operand and its gradient at producer nodes,
                    # except where a consumer's named buffer holds the same elements
                    for u in range(1, cb[1] + 1):
                        covered = False
                        for cn in cons:
                            cc = cfg[cn]
                            if cc[0] == MP and u <= cc[1] and model.consumer_full_T_buffer(I, cn) is not None:
                                covered = True
                            elif cb[0] == DP and cc == cb:
                                covered = True          # the consumer's batch-shard buffer of T
                        if not covered:
                            mem[u - 1] += 2 * V_loc
            else:
                holder = gs.interfaces[p.ref].consumers[0]
                chh = cfg[holder]
                for cn in cons:
                    cc = cfg[cn]
                    t1, ch1 = model.id_fp_pair(I, chh, cc)
                    t2, ch2 = model.id_bp_pair(I, chh, cc)
                    cost += t1 + t2
                    ops += [(f"T{I.idx}", o) for o in ch1 + ch2]
                V_loc = model.local_bytes_input(I, chh)
                cost += model.local_sum_time(V_loc, K + 1)
                # B1 (identity holder): K received contributions + accumulator; the holder's
                # own input-gradient contribution is already inside its layer reservation
                mem[:chh[1]] += (K + 1) * V_loc
        id_holder = None
        for p in I.producers:
            if p.kind == "iface":
                id_holder = gs.interfaces[p.ref].consumers[0]
        for cn in cons:
            cc = cfg[cn]
            V = model.local_bytes_input(I, cc)
            if I.join == "add":
                cost += 3.0 * V / model.cp.beta_red
                if model.shape_preserving_add(I, cn):
                    # J1: first operand summed in place into the consumer's input, second operand
                    # buffered unless it is read in place from an identical holder layout
                    if not (id_holder is not None and cfg[id_holder] == cc):
                        mem[:cc[1]] += V
                else:
                    # J1 with shape-changing post-operators (e.g. ReLU then pooling): both
                    # pre-pooling operands (sum and ReLU in place) and the pre-pooling gradient
                    mem[:cc[1]] += 3 * V
            elif I.consumer_ops.get(cn):
                # C1: consumer-side operators need T and its gradient besides the consumer input
                mem[:cc[1]] += 2 * V
    mem_lb = mem.copy()
    for _, o in ops:
        mem += o.mem()
        mem_lb += o.mem_lower_bound()
    budget = model.budget
    if np.all(mem <= budget + 1e-6):
        status = "feasible"
    elif np.any(mem_lb > budget + 1e-6):
        status = "infeasible"
    else:
        status = "not_established"
    return dict(cost=float(cost), memory=mem, memory_lower_bound=mem_lb, ops=ops, status=status)


__all__ = ["evaluate_graph_plan", "evaluate_graph_plan_full"]
