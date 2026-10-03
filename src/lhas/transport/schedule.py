"""Schedule construction and evaluation (manuscript sec:transport).

A *phase* is a set of ready items (item = one shard, contribution, reduced
tensor or nonzero All-to-all block) with fixed physical endpoints. Routes follow
the shorter circular arc (clockwise on ties) and split every `reach` segments.
Rounds are built by deterministic first-fit (kernels.pack_first_fit).
An *operation* is an ordered list of phases charged independently: it starts
unconfigured, keeps its circuits across local reductions and empty phases, and
skips setup only for a round whose complete circuit set is a subset of the
retained configuration (the set configured at the most recent setup).

Structure (routes, rounds, labels, setup flags) does not depend on payload;
timing and buffer reservations are evaluated per payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..params import TransportParams
from . import kernels as K


# ----------------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------------
def route_geometry(src: np.ndarray, dst: np.ndarray, tp: TransportParams):
    """Return (dirn, D, H, last) for 1-based endpoint arrays.

    Shorter circular arc, clockwise on ties; split at successive `reach`
    segment intervals with a possibly shorter final hop.
    """
    N, L = tp.N, tp.eff_reach
    src = np.asarray(src, np.int64)
    dst = np.asarray(dst, np.int64)
    if np.any(src == dst):
        raise ValueError("self-transfers must be omitted (zero network payload)")
    cw = (dst - src) % N
    ccw = (src - dst) % N
    dirn = np.where(cw <= ccw, 1, -1).astype(np.int8)
    D = np.minimum(cw, ccw).astype(np.int64)
    H = ((D + L - 1) // L).astype(np.int32)
    last = (D - (H - 1) * L).astype(np.int32)
    return dirn, D, H, last


def hop_nodes(src1: int, dirn: int, H: int, last: int, L: int, N: int) -> list[int]:
    """1-based node sequence of a route: src, relays..., dst."""
    nodes = [src1]
    x = src1 - 1
    for k in range(H):
        ns = L if k < H - 1 else last
        x = (x + dirn * ns) % N
        nodes.append(x + 1)
    return nodes


# ----------------------------------------------------------------------------
# data containers
# ----------------------------------------------------------------------------
@dataclass
class Phase:
    name: str
    src: np.ndarray                 # 1-based
    dst: np.ndarray                 # 1-based
    item: np.ndarray                # item identifier (shard id, contributor, block id)
    pay: np.ndarray                 # index into the operation payload vector
    dirn: np.ndarray
    D: np.ndarray
    H: np.ndarray
    last: np.ndarray
    route_round: np.ndarray
    hop_start: np.ndarray
    labels: np.ndarray              # flat per-hop labels (1-based)
    nrounds: int
    order_by_round: np.ndarray
    round_start: np.ndarray
    pipelined: np.ndarray           # uint8 per round
    semantics: str = "transfer"     # transfer | reduce | replace (verification)
    reduce_counts: Optional[np.ndarray] = None   # a_v per node (1-based index-1)
    setup: Optional[np.ndarray] = None           # r flag per round, set by Operation
    joint_note: str = ""
    rule: str = "first_fit"                      # first_fit | dissemination
    rule_meta: dict = field(default_factory=dict)

    @property
    def nroutes(self) -> int:
        return int(self.src.shape[0])


@dataclass
class Operation:
    """One independently charged communication operation (a complete candidate)."""
    op: str                          # 'A2A' | 'AR' | 'AG'
    candidate: str
    N: int
    phases: list[Phase] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def finalize(self, tp: TransportParams) -> "Operation":
        assign_setup_flags(self.phases, tp)
        return self

    @property
    def nrounds(self) -> int:
        return int(sum(p.nrounds for p in self.phases))


# ----------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------
def priority_order(src, dst, item, H, D) -> np.ndarray:
    # decreasing hop count, decreasing total segments, increasing src, dst, item
    return np.lexsort((np.asarray(item), np.asarray(dst), np.asarray(src), -np.asarray(D), -np.asarray(H))).astype(np.int64)


def _group_by_round(route_round: np.ndarray, nrounds: int):
    order = np.argsort(route_round, kind="stable").astype(np.int64)
    counts = np.bincount(route_round, minlength=nrounds)
    start = np.zeros(nrounds + 1, np.int64)
    start[1:] = np.cumsum(counts)
    return order, start


# ----------------------------------------------------------------------------
# structural packing cache
# ----------------------------------------------------------------------------
# Packing depends only on the transport structure (N, labels, budgets, reach,
# peer cap) and on the route list in priority order. When (src, dst) pairs are
# unique, item identifiers cannot break ties and are excluded from the key, so
# e.g. the WRHT top exchange and the one-stage OpTree over the same nodes share
# one packing. The cache never stores payload-dependent quantities.
from collections import OrderedDict

_PACK_CACHE: "OrderedDict" = OrderedDict()
PACK_STATS = {"hits": 0, "misses": 0, "pack_seconds": 0.0, "cached_routes": 0}
PACK_CACHE_MAX_ROUTES = int(__import__("os").environ.get("LHAS_PACK_CACHE_MAX_ROUTES", 3_000_000))  # LRU bound


def _pack_key(src, dst, item, tp: TransportParams) -> bytes:
    import hashlib
    h = hashlib.sha1()
    h.update(repr(tp.structural_key()).encode())
    pair = src * (tp.N + 1) + dst
    h.update(np.ascontiguousarray(pair).tobytes())
    if np.unique(pair).shape[0] != pair.shape[0]:
        h.update(b"items")
        h.update(np.ascontiguousarray(item).tobytes())
    return h.digest()


import os as _os

# Optional persistent cache for expensive packings (payload-independent, exact):
# set LHAS_PACK_CACHE_DIR to a directory; packings with at least
# PACK_DISK_MIN_ROUTES routes are stored as compressed .npz files keyed by the
# structural hash, so interrupted large-N runs resume without repacking.
PACK_DISK_MIN_ROUTES = 50_000


def _disk_path(key: bytes):
    d = _os.environ.get("LHAS_PACK_CACHE_DIR")
    return None if not d else _os.path.join(d, key.hex() + ".npz")


def clear_pack_cache():
    _PACK_CACHE.clear()
    PACK_STATS["cached_routes"] = 0


def _cache_put(key, val):
    _PACK_CACHE[key] = val
    PACK_STATS["cached_routes"] += val[0].shape[0]
    while PACK_STATS["cached_routes"] > PACK_CACHE_MAX_ROUTES and len(_PACK_CACHE) > 1:
        _, old = _PACK_CACHE.popitem(last=False)
        PACK_STATS["cached_routes"] -= old[0].shape[0]


def build_phase(name: str, src, dst, item, tp: TransportParams, pay=None,
                semantics: str = "transfer", reduce_counts=None) -> Phase:
    src = np.asarray(src, np.int64)
    dst = np.asarray(dst, np.int64)
    item = np.asarray(item, np.int64)
    n = src.shape[0]
    pay = np.zeros(n, np.int32) if pay is None else np.asarray(pay, np.int32)
    if n == 0:
        z = np.zeros(0, np.int64)
        return Phase(name, src, dst, item, pay, z.astype(np.int8), z, z.astype(np.int32),
                     z.astype(np.int32), z.astype(np.int32), np.zeros(1, np.int64),
                     np.zeros(0, np.int16), 0, z, np.zeros(1, np.int64), np.zeros(0, np.uint8),
                     semantics, reduce_counts)
    dirn, D, H, last = route_geometry(src, dst, tp)
    hop_start = np.zeros(n + 1, np.int64)
    hop_start[1:] = np.cumsum(H)
    key = _pack_key(src, dst, item, tp)
    hit = _PACK_CACHE.get(key)
    dpath = _disk_path(key) if n >= PACK_DISK_MIN_ROUTES else None
    if hit is None and dpath is not None and _os.path.exists(dpath):
        z = np.load(dpath)
        if z["out_round"].shape[0] == n and z["labels"].shape[0] == int(hop_start[-1]):
            hit = (z["out_round"].astype(np.int32), z["labels"].astype(np.int32), int(z["nr"]))
            PACK_STATS["disk_hits"] = PACK_STATS.get("disk_hits", 0) + 1
            _cache_put(key, hit)
    if hit is not None:
        if key in _PACK_CACHE:
            _PACK_CACHE.move_to_end(key)
        PACK_STATS["hits"] += 1
        out_round, labels, nr = hit[0].copy(), hit[1].copy(), hit[2]
    else:
        import time as _t
        t0 = _t.perf_counter()
        order = priority_order(src, dst, item, H, D)
        out_round = np.full(n, -1, np.int32)
        labels = np.zeros(int(hop_start[-1]), np.int32)
        nr = K.pack_first_fit(order, (src - 1).astype(np.int64), dirn.astype(np.int64), H.astype(np.int64),
                              last.astype(np.int64), tp.eff_reach, tp.N, tp.lam, tp.w_tx, tp.w_rx,
                              tp.d_peer, hop_start, out_round, labels)
        PACK_STATS["misses"] += 1
        PACK_STATS["pack_seconds"] += _t.perf_counter() - t0
        if nr >= 0:
            _cache_put(key, (out_round.copy(), labels.copy(), nr))
            if dpath is not None:
                _os.makedirs(_os.path.dirname(dpath), exist_ok=True)
                tmp = dpath + ".tmp.npz"
                np.savez_compressed(tmp, out_round=out_round.astype(np.int32),
                                    labels=labels.astype(np.int16 if tp.lam < 32767 else np.int32), nr=nr)
                _os.replace(tmp, dpath)
    if nr < 0:
        raise RuntimeError("first-fit made no progress; capacities invalid")
    assert np.all(out_round >= 0)
    ob, rs = _group_by_round(out_round, nr)
    piped = np.zeros(nr, np.uint8)
    np.maximum.at(piped, out_round, (H >= 2).astype(np.uint8))
    return Phase(name, src, dst, item, pay, dirn, D, H, last, out_round, hop_start,
                 labels.astype(np.int16), nr, ob, rs, piped, semantics, reduce_counts)


def build_dissemination(name: str, targets1: Sequence[int], shard_ids: Sequence[int],
                        need: np.ndarray, hold: np.ndarray, tp: TransportParams):
    """E-A dissemination rounds. `hold` is (N x nshards) 0/1 at phase start and
    `need` is (len(targets) x nshards). Ownership updates only after each round.
    Returns (phase, final hold)."""
    N, L = tp.N, tp.eff_reach
    targets0 = np.asarray(targets1, np.int64) - 1
    need = need.astype(np.uint8).copy()
    hold = hold.astype(np.uint8).copy()
    tpos = {int(t): i for i, t in enumerate(targets0)}
    S, Dd, It, Lb, R = [], [], [], [], []
    rnd = 0
    maxper = int(need.sum())
    while need.any():
        cap = max(1, min(maxper, len(targets0) * tp.w_rx))
        os_, od, oi, ol = (np.zeros(cap, np.int64) for _ in range(4))
        n = K.pack_dissemination_round(targets0, need, hold, N, L, tp.lam, tp.w_tx, tp.w_rx,
                                       tp.d_peer, os_, od, oi, ol)
        if n == 0:
            raise RuntimeError("dissemination made no progress (reach graph disconnected?)")
        for q in range(n):
            need[tpos[int(od[q])], oi[q]] = 0
        hold[od[:n], oi[:n]] = 1
        S.append(os_[:n] + 1); Dd.append(od[:n] + 1); It.append(oi[:n]); Lb.append(ol[:n])
        R.append(np.full(n, rnd, np.int32))
        rnd += 1
    if rnd == 0:
        ph = build_phase(name, [], [], [], tp)
        ph.rule = "dissemination"
        ph.rule_meta = dict(targets=[int(t) for t in targets1], shards=[int(x) for x in shard_ids])
        return ph, hold
    src = np.concatenate(S); dst = np.concatenate(Dd)
    idx = np.concatenate(It)
    item = np.asarray(shard_ids, np.int64)[idx]
    dirn, D, H, last = route_geometry(src, dst, tp)
    assert np.all(H == 1)
    rr = np.concatenate(R)
    ob, rs = _group_by_round(rr, rnd)
    hop_start = np.arange(src.shape[0] + 1, dtype=np.int64)
    ph = Phase(name, src, dst, item, np.zeros(src.shape[0], np.int32), dirn, D, H, last, rr,
               hop_start, np.concatenate(Lb).astype(np.int16), rnd, ob, rs,
               np.zeros(rnd, np.uint8), "transfer", None)
    ph.rule = "dissemination"
    ph.rule_meta = dict(targets=[int(t) for t in targets1], shards=[int(x) for x in shard_ids])
    return ph, hold


# ----------------------------------------------------------------------------
# circuit reuse
# ----------------------------------------------------------------------------
def round_circuit_keys(ph: Phase, tp: TransportParams) -> list[np.ndarray]:
    """Sorted unique circuit keys (sender, direction, segments, label) per round."""
    L, N = tp.eff_reach, tp.N
    keys_per_round = []
    if ph.nroutes == 0:
        return keys_per_round
    # expand hops
    r_of_hop = np.repeat(np.arange(ph.nroutes), ph.H)
    k_in_route = np.arange(ph.labels.shape[0]) - ph.hop_start[r_of_hop]
    d = ph.dirn[r_of_hop].astype(np.int64)
    a = (ph.src[r_of_hop] - 1 + d * k_in_route * L) % N
    ns = np.where(k_in_route < ph.H[r_of_hop] - 1, L, ph.last[r_of_hop]).astype(np.int64)
    dirbit = (d < 0).astype(np.int64)
    key = ((a * 2 + dirbit) * (L + 1) + ns) * (tp.lam + 1) + ph.labels.astype(np.int64)
    rr = ph.route_round[r_of_hop]
    o = np.lexsort((key, rr))
    key, rr = key[o], rr[o]
    bounds = np.searchsorted(rr, np.arange(ph.nrounds + 1))
    for q in range(ph.nrounds):
        keys_per_round.append(np.unique(key[bounds[q]:bounds[q + 1]]))
    return keys_per_round


def assign_setup_flags(phases: Sequence[Phase], tp: TransportParams) -> None:
    """Retained-configuration setup rule (sec:transport), numba per round."""
    retained = np.zeros(0, np.int64)
    for ph in phases:
        if ph.nroutes == 0:
            ph.setup = np.ones(0, np.uint8)
            continue
        flags, retained = K.setup_flags_phase(ph.order_by_round, ph.round_start, ph.nrounds,
                                              (ph.src - 1).astype(np.int64), ph.dirn.astype(np.int64),
                                              ph.H.astype(np.int64), ph.last.astype(np.int64),
                                              ph.labels.astype(np.int64), ph.hop_start,
                                              tp.eff_reach, tp.N, tp.lam, retained)
        ph.setup = flags


def assign_setup_flags_reference(phases: Sequence[Phase], tp: TransportParams) -> list:
    """Vectorized reference (memory-heavy for very large phases); tests only."""
    retained = None
    out = []
    for ph in phases:
        flags = np.ones(ph.nrounds, np.uint8)
        for q, keys in enumerate(round_circuit_keys(ph, tp)):
            if retained is not None and keys.size <= retained.size:
                pos = np.searchsorted(retained, keys)
                ok = np.all(pos < retained.size) and np.all(retained[np.minimum(pos, retained.size - 1)] == keys)
                if ok:
                    flags[q] = 0
                    continue
            retained = keys
        out.append(flags)
    return out


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@dataclass
class Evaluation:
    transport: float
    reduction: float
    steps: int
    rounds: int
    setups: int
    relay_peak: np.ndarray          # bytes per node
    gathered: np.ndarray            # bytes per node delivered and retained (AG)
    recv_buffers: np.ndarray        # peak live reduction inputs per node (AR)
    accumulators: np.ndarray        # accumulator bytes per node (AR)
    per_round: Optional[list] = None   # [(setup flags, steps, body)] per phase, if requested

    @property
    def latency(self) -> float:
        return self.transport + self.reduction


def _route_bytes(ph: Phase, payload) -> np.ndarray:
    p = np.asarray(payload, np.int64)
    if p.ndim == 0:
        return np.full(ph.nroutes, int(p), np.int64)
    return p[ph.pay]


def evaluate(op: Operation, payload, tp: TransportParams, beta_red: float,
             accumulator_in_place: bool = False, detail: bool = False) -> Evaluation:
    """Transport latency (Eq. bg_generic_collective), local reduction (Eq.
    local_reduction) and per-node buffer reservations for one payload.

    payload: scalar bytes (uniform items) or a vector indexed by Phase.pay.
    """
    N = tp.N
    transport = 0.0
    reduction = 0.0
    steps = rounds = setups = 0
    relay = np.zeros(N, np.float64)
    gathered = np.zeros(N, np.float64)
    recv = np.zeros(N, np.float64)
    acc = np.zeros(N, np.float64)
    kappa = int(tp.kappa_bytes)
    per_round = [] if detail else None
    for ph in op.phases:
        if detail and not ph.nroutes:
            per_round.append((np.zeros(0, np.uint8), np.zeros(0, np.int64), np.zeros(0)))
        if ph.nroutes:
            nb = _route_bytes(ph, payload)
            if np.any(nb <= 0):
                raise ValueError("zero-byte items must be omitted before scheduling")
            st = np.zeros(ph.nrounds, np.int64)
            body = np.zeros(ph.nrounds, np.float64)
            K.round_body_times(ph.order_by_round, ph.round_start, ph.nrounds, ph.H.astype(np.int64),
                               ph.last.astype(np.int64), nb, tp.eff_reach, kappa, ph.pipelined,
                               tp.b_eff, tp.t_tx, tp.t_rx, tp.t_fwd, tp.seg_time, st, body)
            transport += float(body.sum()) + float(ph.setup.sum()) * tp.t_setup
            if detail:
                per_round.append((ph.setup.copy(), st.copy(), body.copy()))
            steps += int(st.sum()); rounds += ph.nrounds; setups += int(ph.setup.sum())
            K.relay_peaks(ph.order_by_round, ph.round_start, ph.nrounds, (ph.src - 1).astype(np.int64),
                          ph.dirn.astype(np.int64), ph.H.astype(np.int64), nb, tp.eff_reach, N,
                          kappa, ph.pipelined, relay)
            if op.op == "AG":
                np.add.at(gathered, ph.dst - 1, nb)
        if ph.semantics == "reduce" and ph.reduce_counts is not None:
            V = float(np.asarray(payload).reshape(-1)[0]) if np.asarray(payload).ndim else float(payload)
            a = ph.reduce_counts.astype(np.float64)
            if np.any(a > 0):
                reduction += float(np.max((a[a > 0] + 2.0) * V / beta_red))
                recv = np.maximum(recv, a * V)
                if not accumulator_in_place:
                    acc = np.maximum(acc, np.where(a > 0, V, 0.0))
    return Evaluation(transport, reduction, steps, rounds, setups, relay, gathered, recv, acc, per_round)
