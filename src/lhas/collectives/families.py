"""Collective candidate constructions (manuscript Section 2.4 and sec:group_comm).

Every constructor returns one or more `Operation` objects whose phases list the
point-to-point items; packing, timing and buffers are handled by
`lhas.transport.schedule`. Physical node identifiers are preserved throughout;
representative lists are never compressed into a smaller ring.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..params import TransportParams
from ..transport.schedule import Operation, build_phase, build_dissemination


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def prime_factors(n: int) -> list[int]:
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


def mixed_radix_factors(p: int, k: int) -> list[int]:
    """Canonical balanced factorization (sec 2.4, OpTree paragraph): assign prime
    factors in descending order to the bin with the smallest product (ties by bin
    index), then sort radices in nonincreasing order."""
    pf = sorted(prime_factors(p), reverse=True)
    if k < 1 or k > len(pf):
        raise ValueError("depth outside K_p")
    bins = [1] * k
    for q in pf:
        j = min(range(k), key=lambda t: (bins[t], t))
        bins[j] *= q
    out = sorted(bins, reverse=True)
    assert math.prod(out) == p
    return out


def depth_set(p: int) -> list[int]:
    return list(range(1, len(prime_factors(p)) + 1)) if p > 1 else []


def circ(u: int, v: int, N: int) -> int:
    return min((u - v) % N, (v - u) % N)


def nearest(v: int, candidates: Iterable[int], N: int) -> int:
    return min(candidates, key=lambda u: (circ(u, v, N), u))


def mixed_radix_stages(ordered: Sequence[int], radices: Sequence[int]) -> list[list[list[int]]]:
    """Stage j exchanges among list *positions* that differ only in digit j,
    stride n/(m_1...m_j); returns physical-ID subsets per stage."""
    n = len(ordered)
    assert math.prod(radices) == n
    stages, stride = [], n
    for m in radices:
        stride //= m
        subsets = []
        for base in range(n):
            if (base // stride) % m != 0:
                continue
            subsets.append([ordered[base + t * stride] for t in range(m)])
        stages.append(subsets)
    return stages


# ----------------------------------------------------------------------------
# OSM All-to-all (Eq. one-stage): one phase with one item per nonzero entry
# ----------------------------------------------------------------------------
def osm(src: Sequence[int], dst: Sequence[int], tp: TransportParams, name="A2A") -> Operation:
    src = np.asarray(src, np.int64); dst = np.asarray(dst, np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    pay = np.nonzero(keep)[0].astype(np.int32)  # index into the caller's payload vector
    item = (src - 1) * tp.N + (dst - 1)
    op = Operation("A2A", "adapted-OSM", tp.N)
    op.phases.append(build_phase(name, src, dst, item, tp, pay=pay))
    op.meta.update(kept_index=pay)
    return op.finalize(tp)


# ----------------------------------------------------------------------------
# All-gather over an ordered list with (possibly unequal) holdings
# ----------------------------------------------------------------------------
def _stage_items(subsets, hold):
    S, D, I = [], [], []
    for sub in subsets:
        for u in sub:
            for s in sorted(hold[u]):
                for v in sub:
                    if v != u:
                        S.append(u); D.append(v); I.append(s)
    return S, D, I


def _apply_stage(subsets, hold):
    for sub in subsets:
        union = set().union(*(hold[u] for u in sub))
        for u in sub:
            hold[u] = set(union)


def optree_ops(ordered: Sequence[int], tp: TransportParams, hold: Optional[dict] = None,
               tag: str = "OpTree", extra_first_phase=None) -> list[tuple[Operation, dict]]:
    return list(iter_optree_ops(ordered, tp, hold, tag, extra_first_phase))


def iter_optree_ops(ordered: Sequence[int], tp: TransportParams, hold: Optional[dict] = None,
                    tag: str = "OpTree", extra_first_phase=None):
    """Adapted mixed-radix OpTree family: one candidate per depth k in K_n.
    Each node sends each original shard held at the start of a stage to each
    peer. Returns (operation, final holdings) pairs."""
    ordered = list(ordered)
    n = len(ordered)
    hold0 = {u: {u} for u in ordered} if hold is None else {u: set(v) for u, v in hold.items()}
    if n <= 1:
        op = Operation("AG", f"{tag}|k=0", tp.N)
        yield (op.finalize(tp), hold0)
        return
    for k in depth_set(n):
        rad = mixed_radix_factors(n, k)
        h = {u: set(v) for u, v in hold0.items()}
        op = Operation("AG", f"{tag}|k={k}|radix={'x'.join(map(str, rad))}", tp.N)
        for j, subsets in enumerate(mixed_radix_stages(ordered, rad)):
            S, D, I = _stage_items(subsets, h)
            if j == 0 and extra_first_phase is not None:
                S2, D2, I2 = extra_first_phase
                S, D, I = S + list(S2), D + list(D2), I + list(I2)
            op.phases.append(build_phase(f"stage{j+1}", S, D, I, tp))
            _apply_stage(subsets, h)
        op.meta.update(depth=k, radices=rad)
        yield (op.finalize(tp), h)


# ----------------------------------------------------------------------------
# WRHT All-reduce (Eq. bg_wrht_group_constraints, wrht6)
# ----------------------------------------------------------------------------
def wrht_levels(ordered: Sequence[int], m: int):
    levels, L = [], list(ordered)
    while len(L) > m:
        groups = [L[a:a + m] for a in range(0, len(L), m)]
        reps = [g[math.ceil(len(g) / 2) - 1] for g in groups]
        levels.append(list(zip(reps, groups)))
        L = reps
    return levels, L


def wrht_m_values(p: int, tp: TransportParams) -> list[int]:
    hi = min(p, 2 * tp.eff_reach + 1)
    return sorted(set(range(2, hi + 1)) | {p}) if p >= 2 else []


def _counts(N, nodes_counts):
    a = np.zeros(N, np.int64)
    for v, c in nodes_counts:
        a[v - 1] += c
    return a


def wrht_op(ordered: Sequence[int], m: int, mode: str, tp: TransportParams) -> Operation:
    """Collect mode: h collect+reduce phases, then h reverse distributions.
    Exchange mode: h-1 collections, all-to-all of top partial sums with local
    reduction, then h-1 reverse distributions."""
    N = tp.N
    levels, top = wrht_levels(ordered, m)
    op = Operation("AR", f"WRHT|m={m}|{mode}", N)

    def collect(groups, name):
        S, D, I, cnt = [], [], [], []
        for rep, g in groups:
            for u in g:
                if u != rep:
                    S.append(u); D.append(rep); I.append(u)
            if len(g) > 1:
                cnt.append((rep, len(g) - 1))
        op.phases.append(build_phase(name, S, D, I, tp, semantics="reduce",
                                     reduce_counts=_counts(N, cnt)))

    def distribute(groups, name):
        S, D, I = [], [], []
        for rep, g in groups:
            for u in g:
                if u != rep:
                    S.append(rep); D.append(u); I.append(rep)
        op.phases.append(build_phase(name, S, D, I, tp, semantics="replace"))

    for lv, groups in enumerate(levels):
        collect(groups, f"collect{lv+1}")
    if mode == "collect":
        root = top[math.ceil(len(top) / 2) - 1]
        collect([(root, top)], "collect_top")
        distribute([(root, top)], "distribute_top")
    elif mode == "exchange":
        S, D, I = [], [], []
        for u in top:
            for v in top:
                if u != v:
                    S.append(u); D.append(v); I.append(u)
        op.phases.append(build_phase("exchange_top", S, D, I, tp, semantics="reduce",
                                     reduce_counts=_counts(N, [(u, len(top) - 1) for u in top])))
    else:
        raise ValueError(mode)
    for lv in range(len(levels) - 1, -1, -1):
        distribute(levels[lv], f"distribute{lv+1}")
    op.meta.update(m=m, mode=mode, h=len(levels) + 1, top=list(top), levels=levels)
    return op.finalize(tp)


def wrht_ops(ordered: Sequence[int], tp: TransportParams) -> list[Operation]:
    return list(iter_wrht_ops(ordered, tp))


def iter_wrht_ops(ordered: Sequence[int], tp: TransportParams):
    """Lazy WRHT candidates (one operation constructed at a time)."""
    p = len(ordered)
    if p <= 1:
        yield Operation("AR", "WRHT|p=1", tp.N).finalize(tp)
        return
    for m in wrht_m_values(p, tp):
        for mode in ("collect", "exchange"):
            yield wrht_op(ordered, m, mode, tp)


# ----------------------------------------------------------------------------
# All-gather T3(p, q) including unequal groups (sec:group_comm)
# ----------------------------------------------------------------------------
def contraction_group_size(p: int, q: int) -> int:
    """Eq. unequal_groups (p >= 2q)."""
    return p // q if p % q == 0 else q


def allgather_ops(p: int, q: int, tp: TransportParams) -> list[Operation]:
    """All complete candidates delivering all p original shards to each of Q=[1..q]."""
    return list(iter_allgather_ops(p, q, tp))


def iter_allgather_ops(p: int, q: int, tp: TransportParams):
    """Lazy version of allgather_ops: candidates are constructed one at a time
    (large-N memory); the candidate set and order are identical."""
    N = tp.N
    P, Q = list(range(1, p + 1)), list(range(1, q + 1))
    if max(p, q) > N:
        raise ValueError("participants exceed N")
    if p == q:
        for op, _ in iter_optree_ops(P, tp, tag="EQ"):
            yield op
        return
    if p > q:
        # ---------------- C-A (prescribed) ----------------
        if p >= 2 * q:
            g = contraction_group_size(p, q)
            groups = [P[a:a + g] for a in range(0, p, g)]
            reps = [G[0] for G in groups]
            gS, gD, gI = [], [], []
            for G in groups:
                for u in G[1:]:
                    gS.append(u); gD.append(G[0]); gI.append(u)
            hold_after = {u: {u} for u in P}
            for G in groups:
                hold_after[G[0]] = set(G)
            for exch, hold in iter_optree_ops(reps, tp, hold={r: hold_after[r] for r in reps}, tag="rep"):
                op = Operation("AG", f"C-A|g={g}|{exch.candidate}", N)
                op.phases.append(build_phase("gather", gS, gD, gI, tp))
                op.phases.extend(exch.phases)
                h = dict(hold_after); h.update(hold)
                S, D, I = [], [], []
                for v in Q:
                    for s in P:
                        if s not in h.get(v, {v}):
                            src = nearest(v, [r for r in reps if s in h[r]], N)
                            S.append(src); D.append(v); I.append(s)
                op.phases.append(build_phase("deliver", S, D, I, tp))
                op.meta.update(family="C-A", g=g, reps=reps, rep_exchange=exch.candidate)
                yield op.finalize(tp)
        else:
            extra = ([u for u in range(q + 2, p + 1)], [q + 1] * (p - q - 1), [u for u in range(q + 2, p + 1)])
            for tree, hold in iter_optree_ops(Q, tp, tag="Q", extra_first_phase=extra):
                op = Operation("AG", f"C-A|joint|{tree.candidate}", N)
                if not tree.phases and extra[0]:
                    op.phases.append(build_phase("collect", *extra, tp))
                op.phases.extend(tree.phases)
                S, D, I = [], [], []
                for v in Q:
                    for s in range(q + 1, p + 1):
                        S.append(q + 1); D.append(v); I.append(s)
                op.phases.append(build_phase("deliver", S, D, I, tp))
                op.meta.update(family="C-A", q_tree=tree.candidate)
                yield op.finalize(tp)
        # ---------------- C-B superset ----------------
        for tree, _ in iter_optree_ops(P, tp, tag="P"):
            op = Operation("AG", f"C-B|{tree.candidate}", N)
            op.phases = tree.phases
            op.meta.update(family="C-B")
            yield op.finalize(tp)
    else:
        # ---------------- E-A ----------------
        targets = sorted([v for v in Q if v > p], key=lambda v: (min(circ(u, v, N) for u in P), v))
        for tree, _ in iter_optree_ops(P, tp, tag="P"):
            op = Operation("AG", f"E-A|{tree.candidate}", N)
            op.phases = list(tree.phases)
            hold = np.zeros((N, p), np.uint8)
            hold[np.array(P) - 1, :] = 1
            need = np.ones((len(targets), p), np.uint8)
            ph, _ = build_dissemination("disseminate", targets, P, need, hold, tp)
            op.phases.append(ph)
            op.meta.update(family="E-A")
            yield op.finalize(tp)
        # ---------------- E-B superset over Q, empty holdings outside P --------
        hold = {v: ({v} if v <= p else set()) for v in Q}
        for tree, _ in iter_optree_ops(Q, tp, hold=hold, tag="Q"):
            op = Operation("AG", f"E-B|{tree.candidate}", N)
            op.phases = [ph for ph in tree.phases]
            op.meta.update(family="E-B")
            yield op.finalize(tp)
    # ---------------- C-C / E-C direct ----------------
    S, D, I = [], [], []
    for u in P:
        for v in Q:
            if v != u:
                S.append(u); D.append(v); I.append(u)
    op = Operation("AG", ("C-C" if p > q else "E-C") + "|direct", N)
    op.phases.append(build_phase("direct", S, D, I, tp))
    op.meta.update(family="C-C" if p > q else "E-C")
    yield op.finalize(tp)


FAMILY_ORDER = {"EQ": 0, "C-A": 0, "E-A": 0, "C-B": 1, "E-B": 1, "C-C": 2, "E-C": 2}
