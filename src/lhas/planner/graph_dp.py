"""Exact search over typed-interface graphs (GoogLeNet, ResNet-50, and chains).

State at every *major* interface (graph input, multi-consumer fork, concatenation
or addition) is the tuple of its consumers' configurations; branches between
consecutive major interfaces are chains contracted by min-plus products. The
interface step minimizes, for every consumer tuple F', over all producer tuples
(and the identity holder's configuration) with exact lower-bound ordering.

This minimizes the additive objective with each operation at its minimum-charge
alternative; memory feasibility of the resulting plan is checked afterwards
(see lhas.model.branch). No label cap, sampling or time limit is used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ..model.branch import GraphModel, OpChoice
from ..model.chain import DP, MP


@dataclass
class GraphPlan:
    feasible_cost: bool
    cost: float
    configs: dict
    memory: np.ndarray
    memory_feasible: bool
    ops: list
    stats: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        """'feasible': the exact minimizer of the additive objective satisfies the
        corrected reservation budget, hence it is also the memory-constrained optimum
        within the candidate family (D10). Otherwise 'not_established': the
        memory-constrained optimum is not established (this is not a proof that the
        workload is infeasible)."""
        return "feasible" if self.memory_feasible else "not_established"


def _minplus(M, T):
    """(a x b) min-plus (b x c) with argmin."""
    S = M[:, :, None] + T[None, :, :]
    arg = np.argmin(S, axis=1)
    return np.take_along_axis(S, arg[:, None, :], axis=1)[:, 0, :], arg


@njit(cache=True)
def _iface_kernel(Wsorted, Wkey, Wx, Fcfg, qidx, A, P, S, Q, SH, hasH, Hcol, U, out_best, out_arg):
    """For each consumer tuple f (rows of Fcfg), minimize over W entries w:
         W[w] + sum_b (A[b, x_bw, q(f)] + S[b, x_bw] + sum_c P[b, c, x_bw, f_c])
              + hasH * (SH[h_w] + sum_c Q[c, h_w, f_c]) + sum_c U[c, f_c]
    scanning W in ascending lower-bound order Wkey and stopping when the bound
    exceeds the incumbent (exact)."""
    nF, K = Fcfg.shape
    nW, J = Wx.shape
    for fi in range(nF):
        best = np.inf
        arg = -1
        u = 0.0
        for c in range(K):
            u += U[c, Fcfg[fi, c]]
        q = qidx[fi]
        for k in range(nW):
            if Wkey[k] + u >= best:
                break
            v = Wsorted[k] + u
            for b in range(J):
                x = Wx[k, b]
                v += A[b, x, q] + S[b, x]
                for c in range(K):
                    v += P[b, c, x, Fcfg[fi, c]]
                if v >= best:
                    break
            if v >= best:
                continue
            if hasH:
                h = Hcol[k]
                v += SH[h]
                for c in range(K):
                    v += Q[c, h, Fcfg[fi, c]]
            if v < best:
                best = v
                arg = k
        out_best[fi] = best
        out_arg[fi] = arg


def plan_graph(model: GraphModel, verbose: bool = False) -> GraphPlan:
    t0 = time.perf_counter()
    gs = model.gs
    I_all = gs.interfaces
    major = [I.idx for I in I_all if I.join in ("input", "cat", "add") or len(I.consumers) > 1 or I.identity_users]
    is_major = set(major)
    A = {n: model.configs(n) for n in model.order}
    if any(len(v) == 0 for v in A.values()):
        return GraphPlan(False, float("inf"), {}, np.zeros(model.N), False, [], dict(reason="empty configuration set"))
    stats = dict(major_interfaces=len(major), configs={n: len(v) for n, v in A.items()})

    def branch_of(f):
        chain, cur = [f], f
        while True:
            oi = gs.iface_of_output.get(cur)
            if oi is None or oi in is_major:
                return chain, oi
            nxt = I_all[oi].consumers
            assert len(nxt) == 1
            cur = nxt[0]
            chain.append(cur)

    # --- chain (minor interface) transition matrix between consecutive branch layers
    def minor_T(Iidx, a, b):
        I = I_all[Iidx]
        prod = I.producers[0]
        T = np.full((len(A[a]), len(A[b])), np.inf)
        for i, ca in enumerate(A[a]):
            for j, cb in enumerate(A[b]):
                if ca[0] == MP:
                    g, _ = model.charge(a, ca, (ca[1], cb[1], (model.psi * model.rows(I, prod) * model.B) // ca[1], None))
                else:
                    g, _ = model.charge(a, ca)
                    g += model.fp_pair_dp(I, prod, ca, cb)[0]
                g += model.bp_pair(I, prod, ca, cb)[0]
                T[i, j] = g
        return T

    # V: (axes names, array); start at the input interface
    first = I_all[major[0]]
    assert first.join == "input"
    axes = list(first.consumers)
    V = np.zeros(tuple(len(A[c]) for c in axes))
    back = []           # per segment: dict of backtracking information
    k = 0
    final_layer = gs.output_layer
    while True:
        M = I_all[major[k]]
        # ------------------------------------------------ branch contraction
        chains = {}
        nxt_major = None
        for f in M.consumers:
            ch, oi = branch_of(f)
            chains[f] = ch
            if nxt_major is None:
                nxt_major = oi
            elif oi != nxt_major:
                raise ValueError("graph is not series-parallel between major interfaces")
        nextI = I_all[nxt_major] if nxt_major is not None else None
        holder = None
        if nextI is not None:
            for p in nextI.producers:
                if p.kind == "iface":
                    if p.ref != M.idx:
                        raise ValueError("identity operand does not come from the previous major interface")
                    holder = M.consumers[0]
        seg = dict(M=M.idx, chains=chains, holder=holder, contract={})
        W = V
        waxes = list(axes)
        for f in M.consumers:
            ch = chains[f]
            Cb = np.zeros((len(A[f]), len(A[f])))
            Cb[:] = np.inf
            np.fill_diagonal(Cb, 0.0)
            args = []
            for j in range(len(ch) - 1):
                Iidx = gs.iface_of_output[ch[j]]
                Cb, arg = _minplus(Cb, minor_T(Iidx, ch[j], ch[j + 1]))
                args.append(arg)
            x = ch[-1]
            ax = waxes.index(f)
            Wm = np.moveaxis(W, ax, -1)                       # [..., f]
            if f == holder:
                Wn = Wm[..., :, None] + Cb                     # [..., f, x]  keep holder axis
                W = Wn
                waxes = [a for a in waxes if a != f] + [f, "x:" + x]
                seg["contract"][f] = ("keep", args, Cb)
            else:
                S = Wm[..., :, None] + Cb                      # [..., f, x]
                argf = np.argmin(S, axis=-2)
                W = np.take_along_axis(S, argf[..., None, :], axis=-2)[..., 0, :]
                waxes = [a for a in waxes if a != f] + ["x:" + x]
                seg["contract"][f] = ("min", args, argf, list(waxes))
        # ------------------------------------------------ terminal
        if nextI is None:
            assert len(M.consumers) == 1 and chains[M.consumers[0]][-1] == final_layer
            x = final_layer
            ax = waxes.index("x:" + x)
            term = np.zeros(len(A[x]))
            for i, c in enumerate(A[x]):
                if c[0] == MP:
                    L = model.L[x]
                    mask = np.ones(model.N); mask[:c[1]] = 0.0
                    term[i] = model.charge(x, c, (c[1], c[1], (model.psi * L.y_numel * model.B) // c[1], mask))[0]
                else:
                    term[i] = model.charge(x, c)[0]
            Wm = np.moveaxis(W, ax, -1)
            tot = Wm + term
            flat = int(np.argmin(tot))
            cost = float(tot.reshape(-1)[flat])
            seg["terminal"] = dict(axes=[a for a in waxes if a != "x:" + x] + ["x:" + x], shape=tot.shape, flat=flat)
            back.append(seg)
            break
        # ------------------------------------------------ interface step
        cons = list(nextI.consumers)
        K = len(cons)
        prods = []
        for p in nextI.producers:
            if p.kind == "layer":
                prods.append(p)
        # W axes: producers x_b (+ holder)
        for p in prods:
            assert "x:" + p.ref in waxes
        order_axes = ["x:" + p.ref for p in prods] + ([holder] if holder is not None else [])
        if set(order_axes) != set(waxes):
            raise ValueError(f"unexpected remaining axes {waxes} vs {order_axes}")
        W = np.transpose(W, [waxes.index(a) for a in order_axes])
        Wflat = W.reshape(-1)
        idx = np.array(np.unravel_index(np.arange(Wflat.size), W.shape)).T.astype(np.int64)
        J = len(prods)
        Wx = np.ascontiguousarray(idx[:, :J]) if J else np.zeros((Wflat.size, 1), np.int64)
        Hcol = np.ascontiguousarray(idx[:, J]) if holder is not None else np.zeros(Wflat.size, np.int64)
        # consumer tuples
        Fdims = [len(A[c]) for c in cons]
        maxA = max(max(Fdims), max((len(A[p.ref]) for p in prods), default=1),
                   len(A[holder]) if holder is not None else 1)
        qvals = sorted({c[1] for cn in cons for c in A[cn]})
        qpos = {q: i for i, q in enumerate(qvals)}
        Fcfg = np.array(list(np.ndindex(*Fdims)), dtype=np.int64).reshape(-1, K)
        qidx = np.array([qpos[max(A[cons[c]][Fcfg[f, c]][1] for c in range(K))] for f in range(Fcfg.shape[0])],
                        dtype=np.int64)
        Jn = max(J, 1)
        Aarr = np.zeros((Jn, maxA, len(qvals)))
        Parr = np.zeros((Jn, K, maxA, maxA))
        Sarr = np.zeros((Jn, maxA))
        for b, p in enumerate(prods):
            xn = p.ref
            for i, cb in enumerate(A[xn]):
                V_loc = model.local_bytes_layer(nextI, p, cb)
                Sarr[b, i] = model.local_sum_time(V_loc, K)
                if cb[0] == MP:
                    for qi, q in enumerate(qvals):
                        Aarr[b, i, qi] = model.charge(xn, cb, (cb[1], q, (model.psi * model.rows(nextI, p) * model.B) // cb[1], None))[0]
                else:
                    Aarr[b, i, :] = model.charge(xn, cb)[0]
                for c, cn in enumerate(cons):
                    for j, cc in enumerate(A[cn]):
                        v = model.bp_pair(nextI, p, cb, cc)[0]
                        if cb[0] == DP:
                            v += model.fp_pair_dp(nextI, p, cb, cc)[0]
                        Parr[b, c, i, j] = v
        Qarr = np.zeros((K, maxA, maxA))
        SH = np.zeros(maxA)
        if holder is not None:
            for h, chh in enumerate(A[holder]):
                SH[h] = model.local_sum_time(model.local_bytes_input(nextI, chh), K + 1)
                for c, cn in enumerate(cons):
                    for j, cc in enumerate(A[cn]):
                        Qarr[c, h, j] = model.id_fp_pair(nextI, chh, cc)[0] + model.id_bp_pair(nextI, chh, cc)[0]
        U = np.zeros((K, maxA))
        if nextI.join == "add":
            for c, cn in enumerate(cons):
                for j, cc in enumerate(A[cn]):
                    U[c, j] = 3.0 * model.local_bytes_input(nextI, cc) / model.cp.beta_red
        # lower-bound ordering of W entries
        lb = Wflat.copy()
        for b in range(J):
            mins = (Aarr[b].min(axis=1) + Sarr[b] + Parr[b].min(axis=2).sum(axis=0))[: len(A[prods[b].ref])]
            lb = lb + mins[Wx[:, b]]
        if holder is not None:
            hmin = SH[: len(A[holder])] + Qarr[:, : len(A[holder]), :].min(axis=2).sum(axis=0)
            lb = lb + hmin[Hcol]
        fin = np.isfinite(lb)
        order = np.argsort(np.where(fin, lb, np.inf), kind="stable")
        order = order[fin[order]]
        best = np.zeros(Fcfg.shape[0]); arg = np.zeros(Fcfg.shape[0], np.int64)
        tk = time.perf_counter()
        _iface_kernel(np.ascontiguousarray(Wflat[order]), np.ascontiguousarray(lb[order]),
                      np.ascontiguousarray(Wx[order]), Fcfg, qidx, Aarr, Parr, Sarr, Qarr, SH,
                      holder is not None, np.ascontiguousarray(Hcol[order]), U, best, arg)
        if verbose:
            print(f"  interface T{nextI.idx} ({nextI.join}): |F'|={Fcfg.shape[0]} |W|={Wflat.size} "
                  f"{time.perf_counter() - tk:.2f}s", flush=True)
        seg["iface"] = dict(I=nextI.idx, order_axes=order_axes, Wshape=W.shape, order=order, arg=arg,
                            cons=cons, Fdims=Fdims)
        back.append(seg)
        V = best.reshape(Fdims)
        axes = cons
        k = major.index(nextI.idx)
    stats["search_seconds"] = time.perf_counter() - t0
    stats["search_cost"] = cost
    # ------------------------------------------------ backtracking
    cfg = {}
    seg = back[-1]
    t = seg["terminal"]
    vals = np.unravel_index(t["flat"], t["shape"])
    cur = dict(zip(t["axes"], vals))
    for si in range(len(back) - 1, -1, -1):
        seg = back[si]
        # cur holds indices of this segment's post-contraction axes; undo contractions
        for f in reversed(list(seg["contract"].keys())):
            info = seg["contract"][f]
            ch = seg["chains"][f]
            x = ch[-1]
            xi = cur["x:" + x]
            if info[0] == "keep":
                fi = cur[f]
            else:
                argf, axes_after = info[2], info[3]
                sub = tuple(cur[a] for a in axes_after)
                fi = int(argf[sub])
                cur[f] = fi
            # recover internal chain configurations
            idxs = [None] * len(ch)
            idxs[-1] = xi
            for j in range(len(ch) - 2, -1, -1):
                arg = info[1][j]
                idxs[j] = int(arg[fi, idxs[j + 1]]) if j > 0 else fi
            idxs[0] = fi
            for j, n in enumerate(ch):
                cfg[n] = A[n][idxs[j]]
            cur.pop("x:" + x, None)
        if si == 0:
            break
        prev = back[si - 1]
        pI = prev["iface"]
        Fidx = tuple(cur[c] for c in pI["cons"])
        flatF = int(np.ravel_multi_index(Fidx, pI["Fdims"]))
        w = int(pI["order"][pI["arg"][flatF]])
        widx = np.unravel_index(w, pI["Wshape"])
        cur = dict(zip(pI["order_axes"], widx))
    return _finish(model, cfg, stats)


def _finish(model: GraphModel, cfg: dict, stats: dict) -> GraphPlan:
    """Recompute the plan's cost and reservation vector from the configurations."""
    from .graph_eval import evaluate_graph_plan_full
    r = evaluate_graph_plan_full(model, cfg)
    cost, mem, ops = r["cost"], r["memory"], r["ops"]
    stats["evaluated_cost"] = cost
    sc = stats["search_cost"]
    if abs(sc - cost) > 1e-9 * max(1.0, cost):
        raise AssertionError(f"search objective {sc} differs from re-evaluated plan cost {cost}")
    return GraphPlan(True, cost, cfg, mem, bool(np.all(mem <= model.budget + 1e-6)), ops, stats)


__all__ = ["plan_graph", "GraphPlan"]
