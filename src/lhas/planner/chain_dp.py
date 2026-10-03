"""Memory-aware chain dynamic program (manuscript sec:dp, Algorithm alg:optical-config).

State: (configuration c_i, accumulated reservation vector mu) with cost D_i.
Transitions enumerate every nondominated schedule combination of the layer and
boundary operations (lhas.model.chain). A label is dropped when it violates the
capacity, or when another label for the same configuration has no greater cost
and no greater reservation on any node (Eq. memory_dp_recurrence).

Additional exact pruning (documented in docs/verification.md, "Pruning rules"): a label is *safe* when
its reservation plus the largest possible future reservation fits on every
node; every completion of a safe label is feasible. For each configuration the
cheapest safe label therefore dominates every label with no smaller cost, safe
or not. When every label is safe from the start (the memory constraint cannot
bind for any plan), this reduces to the unconstrained recurrence (Eq. optimal6)
with the minimum-charge combination per transition. Both reductions preserve the
optimum of the represented objective; they are checked against exhaustive
enumeration in tests/test_planner_exhaustive.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ..model.chain import ChainModel

import os
_DEBUG = bool(os.environ.get("LHAS_DEBUG"))


class LabelLimitExceeded(RuntimeError):
    """Raised (never silently) when an exact label search exceeds a declared cap."""


@dataclass
class PlanResult:
    feasible: bool
    cost: float
    configs: list
    choices: list
    memory: np.ndarray
    breakdown: dict
    stats: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        """'feasible' (a plan satisfying the reservation budget, optimal for the searched
        space), 'infeasible' (exact search completed without a feasible plan), or
        'not_established' (search stopped at a declared resource limit)."""
        if self.feasible:
            return "feasible"
        if self.stats.get("unavailable"):
            return "unavailable"           # a required compute input is missing (profiled mode)
        return "not_established" if self.stats.get("not_completed") else "infeasible"


@njit(cache=True)
def _pareto_kernel(order, D, M):
    n, d = M.shape
    keep = np.empty(n, np.int64)
    nk = 0
    for ii in range(n):
        i = order[ii]
        dom = False
        for kk in range(nk):
            j = keep[kk]
            if D[j] <= D[i] + 1e-15:
                ok = True
                for v in range(d):
                    if M[j, v] > M[i, v] + 1e-6:
                        ok = False
                        break
                if ok:
                    dom = True
                    break
        if not dom:
            keep[nk] = i
            nk += 1
    return keep[:nk]


def _pareto_labels(D, M, future=None, budget=None):
    """Indices of labels not dominated in (cost, reservation vector).

    With `future` (largest possible future reservation per node) a coordinate on
    which a label is safe (mu_v + future_v <= budget) is replaced by 0: every
    completion is feasible there, so a label that is safe on v is at least as
    good on v as any label. Coordinates on which all labels are safe are dropped.
    """
    if D.shape[0] <= 1:
        return np.arange(D.shape[0], dtype=np.int64)
    Mt = M
    if future is not None:
        unsafe = (M + future[None, :]) > budget + 1e-6
        cols = np.nonzero(unsafe.any(axis=0))[0]
        Mt = np.where(unsafe[:, cols], M[:, cols], 0.0)
    Mt = np.ascontiguousarray(Mt, dtype=np.float64)
    order = np.lexsort((Mt.sum(axis=1), D)).astype(np.int64)
    keep = _pareto_kernel(order, np.ascontiguousarray(D, dtype=np.float64), Mt)
    return np.sort(keep)


def plan_chain(model: ChainModel, force_labels: bool = False, pareto_alts: bool = True,
               beam: int = 64, max_labels: int | None = None) -> PlanResult:
    t0 = time.perf_counter()
    Lc = len(model.L)
    alpha = model.st.alpha
    budget = model.budget
    N = model.N
    A = [model.configs(i) for i in range(Lc)]
    stats = dict(configs=[len(a) for a in A])
    if any(len(a) == 0 for a in A):
        return PlanResult(False, float("inf"), [], [], np.zeros(N), {}, dict(stats, reason="empty A_i"))
    mvec = [{c: model.m(i, c) for c in A[i]} for i in range(Lc)]
    tcp = [{c: model.tcp(i, c) for c in A[i]} for i in range(Lc)]
    # transition tables: (G array, E matrix, alternatives)
    trans = []
    for i in range(Lc - 1):
        d = {}
        for ci in A[i]:
            for cj in A[i + 1]:
                alts = model.transition(i, ci, cj, pareto=pareto_alts)
                d[(ci, cj)] = (np.array([a.G(tcp[i][ci], alpha) for a in alts]), np.stack([a.e for a in alts]), alts)
        trans.append(d)
    term = {}
    for c in A[-1]:
        alts = model.terminal(c, pareto=pareto_alts)
        term[c] = (np.array([a.G(tcp[-1][c], alpha) for a in alts]), np.stack([a.e for a in alts]), alts)
    stats["alternatives"] = int(sum(len(v[2]) for d in trans for v in d.values()) + sum(len(v[2]) for v in term.values()))
    # admissible bounds ignoring memory: cost-to-go h and cost-to-come f
    h = [dict() for _ in range(Lc)]
    for c in A[-1]:
        h[-1][c] = float(term[c][0].min())
    for i in range(Lc - 2, -1, -1):
        for ci in A[i]:
            h[i][ci] = min(float(trans[i][(ci, cj)][0].min()) + h[i + 1][cj] for cj in A[i + 1])
    f = [dict() for _ in range(Lc)]
    for c in A[0]:
        f[0][c] = 0.0
    for i in range(Lc - 1):
        for cj in A[i + 1]:
            f[i + 1][cj] = min(f[i][ci] + float(trans[i][(ci, cj)][0].min()) for ci in A[i])
    lb = min(h[0].values())
    stats["unconstrained_optimum"] = lb

    def future_bound(UB):
        """Largest reservation still addable after a label at layer k, restricted
        to configurations and alternatives that can occur in a plan of cost <= UB."""
        tol = UB * (1 + 1e-12)
        Mx, Ex = [], []
        for j in range(Lc):
            vs = [mvec[j][c] for c in A[j] if f[j][c] + h[j][c] <= tol]
            Mx.append(np.max(np.stack(vs), axis=0) if vs else np.zeros(N))
        for i in range(Lc - 1):
            e = np.zeros(N)
            for (ci, cj), (G, E, _) in trans[i].items():
                ok = f[i][ci] + G + h[i + 1][cj] <= tol
                if ok.any():
                    np.maximum(e, E[ok].max(axis=0), out=e)
            Ex.append(e)
        et = np.zeros(N)
        for c, (G, E, _) in term.items():
            ok = f[-1][c] + G <= tol
            if ok.any():
                np.maximum(et, E[ok].max(axis=0), out=et)
        fut = [None] * Lc
        acc = et.copy()
        for k in range(Lc - 1, -1, -1):
            fut[k] = acc.copy()
            if k > 0:
                acc = acc + Mx[k] + Ex[k - 1]
        return fut, bool(np.all(Mx[0] + fut[0] <= budget + 1e-6))

    # lower bound on reservations that every completion must still add (exact pruning)
    Mmin = [np.min(np.stack([mvec[j][c] for c in A[j]]), axis=0) for j in range(Lc)]
    Emin = [np.min(np.concatenate([E for (_, E, _) in trans[i].values()]), axis=0) for i in range(Lc - 1)]
    Emin_t = np.min(np.concatenate([E for (_, E, _) in term.values()]), axis=0)
    fmin = [None] * Lc
    acc_min = Emin_t.copy()
    for k in range(Lc - 1, -1, -1):
        fmin[k] = acc_min.copy()
        if k > 0:
            acc_min = acc_min + Mmin[k] + Emin[k - 1]

    def run(use_labels, UB=float("inf"), beam_w=None, fut=None):
        tol = UB * (1 + 1e-12)
        labels = [dict() for _ in range(Lc)]
        for c in A[0]:
            if np.all(mvec[0][c] + fmin[0] <= budget + 1e-6) and h[0][c] <= tol:
                labels[0][c] = dict(D=np.array([0.0]), M=mvec[0][c][None, :].copy(), back=[None])
        hist = [sum(len(v["D"]) for v in labels[0].values())]
        for i in range(Lc - 1):
            nxt = {}
            for cj in A[i + 1]:
                Dp, Mp, backs = [], [], []
                parts, total = [], 0
                for ci, lab in labels[i].items():
                    G, E, alts = trans[i][(ci, cj)]
                    if not use_labels:
                        k = int(np.argmin(G)); G, E, alts = G[k:k + 1], E[k:k + 1], [alts[k]]
                    Dc = lab["D"][:, None] + G[None, :]
                    li, ai = np.nonzero(Dc + h[i + 1][cj] <= tol)
                    if li.size == 0:
                        continue
                    parts.append((ci, lab, E, alts, li, ai, Dc[li, ai]))
                    total += li.size
                    # the cap is checked before any reservation matrix is materialized
                    if max_labels is not None and use_labels and beam_w is None and total > max_labels:
                        err = LabelLimitExceeded(f"layer {i+1} config {cj}: more than {max_labels} candidate "
                                                 "labels; exact search not completed")
                        # bounds known when the search stops: the unconstrained optimum is a lower
                        # bound; a beam incumbent (if any) is the cost of a feasible plan
                        err.lower_bound = lb
                        err.incumbent = stats.get("incumbent")
                        raise err
                if _DEBUG:
                    print(f"  layer {i+1} cfg {cj}: candidates {total} (labels={use_labels}, UB={UB:.6g})", flush=True)
                mj = mvec[i + 1][cj][None, :]
                for (ci, lab, E, alts, li, ai, Dsel) in parts:
                    for s0 in range(0, li.size, 4096):          # bounded working memory
                        l_, a_ = li[s0:s0 + 4096], ai[s0:s0 + 4096]
                        Mc = lab["M"][l_] + E[a_] + mj
                        ok = np.all(Mc + fmin[i + 1] <= budget + 1e-6, axis=1)
                        if ok.any():
                            Dp.append(Dsel[s0:s0 + 4096][ok])
                            Mp.append(Mc[ok])
                            backs += [(ci, int(x), alts[int(y)]) for x, y in zip(l_[ok], a_[ok])]
                if not Dp:
                    continue
                D = np.concatenate(Dp); M = np.concatenate(Mp)
                if use_labels:
                    safe = np.all(M + fut[i + 1] <= budget + 1e-6, axis=1)
                    if safe.any():
                        s_best = np.nonzero(safe)[0][np.argmin(D[safe])]
                        keep = D < D[s_best] - 1e-15
                        keep[s_best] = True
                        k2 = np.nonzero(keep)[0]
                        D, M = D[k2], M[k2]; backs = [backs[k] for k in k2]
                    keep = _pareto_labels(D, M, fut[i + 1], budget)
                    if beam_w is not None and keep.size > beam_w:
                        by_cost = keep[np.argsort(D[keep], kind="stable")[: beam_w // 2]]
                        by_mem = keep[np.argsort(M[keep].max(axis=1), kind="stable")[: beam_w - beam_w // 2]]
                        keep = np.unique(np.concatenate([by_cost, by_mem]))
                else:
                    keep = np.array([int(np.argmin(D))])
                nxt[cj] = dict(D=D[keep], M=M[keep], back=[backs[k] for k in keep])
            labels[i + 1] = nxt
            hist.append(sum(len(v["D"]) for v in nxt.values()))
        best = None
        for c, lab in labels[-1].items():
            G, E, alts = term[c]
            if not use_labels:
                k = int(np.argmin(G)); G, E, alts = G[k:k + 1], E[k:k + 1], [alts[k]]
            for li in range(lab["D"].shape[0]):
                for g, e, a in zip(G, E, alts):
                    tot = lab["D"][li] + g
                    mu = lab["M"][li] + e
                    if np.all(mu <= budget + 1e-6) and (best is None or tot < best[0] - 1e-15):
                        best = (tot, c, li, a, mu)
        return best, labels, hist

    # 1) unconstrained recurrence; exact when its minimizer is memory-feasible
    fut_all, slack = future_bound(float("inf"))
    stats["memory_slack"] = slack
    best, labels, hist = run(False)
    if not force_labels and (slack or (best is not None and abs(best[0] - lb) <= 1e-12 * max(1.0, lb))):
        stats["mode"] = ("cost-only (memory provably non-binding)" if slack
                         else "cost-only minimizer is memory-feasible (equals lower bound)")
    else:
        # 2) incumbent (upper bound) from a bounded-width label search: heuristic
        inc = best
        UB = float("inf") if best is None else best[0]
        if beam:
            fut_b, _ = future_bound(UB)
            b2, _, _ = run(True, UB=UB, beam_w=beam, fut=fut_b)
            if b2 is not None and (inc is None or b2[0] < inc[0]):
                inc, UB = b2, b2[0]
        stats["incumbent"] = UB
        # 3) exact label search with branch-and-bound on the admissible bound
        fut_ub, _ = future_bound(UB)
        best, labels, hist = run(True, UB=UB, fut=fut_ub)
        if inc is not None and (best is None or inc[0] < best[0] - 1e-15):
            raise AssertionError("exact search lost the incumbent; bounds inconsistent")
        stats["mode"] = "labels (exact, branch-and-bound)"
    stats["labels_per_layer"] = hist
    stats["seconds"] = time.perf_counter() - t0
    if best is None:
        return PlanResult(False, float("inf"), [], [], np.zeros(N), {}, dict(stats, reason="no feasible plan"))
    tot, c, li, a_term, mu = best
    configs, choices = [c], [a_term]
    for i in range(Lc - 1, 0, -1):
        ci, li_prev, a = labels[i][c]["back"][li]
        configs.append(ci); choices.append(a)
        c, li = ci, li_prev
    configs.reverse(); choices.reverse()
    # breakdown (computation, intra transport, intra reduction, inter, overlap credit)
    bd = dict(compute=0.0, intra_transport=0.0, intra_reduction=0.0, inter=0.0, overlap_credit=0.0)
    for i, (c, a) in enumerate(zip(configs, choices)):
        t = tcp[i][c]
        bd["compute"] += t
        bd["intra_transport"] += a.tr_intra
        bd["intra_reduction"] += a.red_intra
        bd["inter"] += a.inter
        bd["overlap_credit"] += alpha * min(t, a.tr_intra)
    assert abs(sum(bd[k] for k in ("compute", "intra_transport", "intra_reduction", "inter"))
               - bd["overlap_credit"] - tot) <= 1e-9 * max(1.0, tot)
    return PlanResult(True, float(tot), configs, choices, mu, bd, stats)


def exhaustive_chain(model: ChainModel, pareto_alts: bool = False):
    """Brute force over every configuration sequence and every schedule
    combination (tests only; exponential)."""
    import itertools
    Lc = len(model.L)
    A = [model.configs(i, memory_filter=False) for i in range(Lc)]
    best = (float("inf"), None)
    for seq in itertools.product(*A):
        base = sum(model.m(i, c) for i, c in enumerate(seq))
        opts = [[(a.G(model.tcp(i, seq[i]), model.st.alpha), a.e)
                 for a in model.transition(i, seq[i], seq[i + 1], pareto=pareto_alts)] for i in range(Lc - 1)]
        opts.append([(a.G(model.tcp(Lc - 1, seq[-1]), model.st.alpha), a.e)
                     for a in model.terminal(seq[-1], pareto=pareto_alts)])
        for combo in itertools.product(*opts):
            mu = base + sum(e for _, e in combo)
            if np.all(mu <= model.budget + 1e-6):
                cost = sum(g for g, _ in combo)
                if cost < best[0] - 1e-15:
                    best = (cost, seq)
    return best


__all__ = ["plan_chain", "exhaustive_chain", "PlanResult", "LabelLimitExceeded"]
