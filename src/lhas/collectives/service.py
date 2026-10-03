"""Collective alternatives with latency parts and per-node reservation ledgers.

`CollectiveService` constructs every candidate of the requested collective once
per structure (participants, holdings, transport parameters), evaluates it for
each payload, and returns the nondominated alternatives. An alternative keeps

  transport   optical transport time (Eq. bg_generic_collective), setups included
  reduction   local-reduction time (Eq. local_reduction)
  ledger      per-node byte vectors, one per reservation category:
                relay      2 min(kappa, V) per concurrently reserved relayed route,
                           peak over rounds (reusable after each round)
                retained   received All-reduce contributions live until the
                           phase's local reduction completes, peak over phases
                accum      reduction output accumulator (not in place), peak over phases
                assembled  All-gather shards received and retained (the caller
                           decides which of these alias an existing reservation)

Dominance is taken componentwise over (transport, reduction, every ledger
category on every node); a dominated alternative can never yield a cheaper or
lighter plan under any nonnegative aliasing mask or overlap alpha in [0, 1].
Exact duplicates keep the alternative preferred by the manuscript tie rule
(fewer steps, fewer rounds, then family A, B, C, then smaller depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..params import TransportParams, ComputeParams
from ..transport.schedule import evaluate, Operation
from . import families as F

CATS = ("relay", "retained", "accum", "assembled")


@dataclass
class Alt:
    op: str
    cand: str
    transport: float
    reduction: float
    ledger: dict
    steps: int = 0
    rounds: int = 0
    setups: int = 0

    @property
    def latency(self) -> float:
        return self.transport + self.reduction

    def mem(self, assembled_mask=None) -> np.ndarray:
        m = self.ledger["relay"] + self.ledger["retained"] + self.ledger["accum"]
        a = self.ledger["assembled"]
        return m + (a if assembled_mask is None else a * assembled_mask)


def _zero_alt(op, N, name):
    z = np.zeros(N)
    return Alt(op, name, 0.0, 0.0, {c: z.copy() for c in CATS})


def _tie_key(op: Operation, ev):
    fam = op.candidate.split("|")[0]
    depth = op.meta.get("depth", 0) if isinstance(op.meta, dict) else 0
    return (ev.latency, ev.steps, ev.rounds, F.FAMILY_ORDER.get(fam, 0), depth, op.candidate)


def pareto(alts: list) -> list:
    """Remove alternatives dominated componentwise; input order breaks exact ties."""
    if len(alts) <= 1:
        return list(alts)
    keys = np.array([[a.transport, a.reduction] for a in alts])
    mats = np.stack([np.concatenate([a.ledger[c] for c in CATS]) for a in alts])
    keep = []
    for i in range(len(alts)):
        dom = False
        for j in keep:
            if np.all(keys[j] <= keys[i] + 1e-15) and np.all(mats[j] <= mats[i] + 1e-9):
                dom = True
                break
        if dom:
            continue
        # remove previously kept alternatives strictly dominated by i
        keep = [j for j in keep if not (np.all(keys[i] <= keys[j] + 1e-15) and np.all(mats[i] <= mats[j] + 1e-9))]
        keep.append(i)
    return [alts[i] for i in sorted(keep)]


@dataclass
class CollectiveService:
    tp: TransportParams
    cp: ComputeParams
    _ar: dict = field(default_factory=dict)
    _ag: dict = field(default_factory=dict)
    _a2a: dict = field(default_factory=dict)
    _res: dict = field(default_factory=dict)
    keep_structures: bool = True           # False: drop candidate structures after each evaluation
    family: str = "all"                    # collective comparison: 'all' | 'one-stage' | 'deepest'

    stats: dict = field(default_factory=lambda: {"built_ar": 0, "built_ag": 0, "built_a2a": 0})

    # ------------------------------------------------------------------ structures
    def ar_ops(self, p: int):
        if p not in self._ar:
            self._ar[p] = F.wrht_ops(list(range(1, p + 1)), self.tp)
            self.stats["built_ar"] += 1
        return self._ar[p]

    def ag_ops(self, p: int, q: int):
        if (p, q) not in self._ag:
            self._ag[(p, q)] = [] if (p == 1 and q == 1) else F.allgather_ops(p, q, self.tp)
            self.stats["built_ag"] += 1
        return self._ag[(p, q)]

    def release(self):
        self._ar.clear(); self._ag.clear(); self._a2a.clear()

    def prefetch(self, demands, log=None) -> dict:
        """Evaluate demands grouped by structure and ordered so that structures
        sharing route sets (P- and Q-trees across receivers, all-pairs sets used
        by All-gather and All-to-all) are packed while still in the LRU packing
        cache. demands: ('AR', p, V) | ('AG', p, q, V) | ('A2A', key, src, dst, nbytes)."""
        import time as _t
        ar, ag, a2a = {}, {}, {}
        for d in demands:
            if d[0] == "AR":
                ar.setdefault(d[1], set()).add(int(d[2]))
            elif d[0] == "AG":
                ag.setdefault((d[1], d[2]), set()).add(int(d[3]))
            else:
                a2a[d[1]] = d[2:]
        st = {"ar_groups": len(ar), "ag_groups": len(ag), "a2a": len(a2a), "seconds": {"AR": 0.0, "AG": 0.0, "A2A": 0.0}}
        items = [("AR", (p, p), p) for p in ar] + [("AG", k, k) for k in ag]
        for key, (s_, d_, b_) in a2a.items():
            pq = (int(s_.max()) if s_.size else 0, int(d_.max()) if d_.size else 0)
            items.append(("A2A", pq, key))
        items.sort(key=lambda it: (it[1][0], it[1][1], {"AR": 0, "AG": 1, "A2A": 2}[it[0]]))
        done = 0
        for kind, pq, key in items:
            t0 = _t.perf_counter()
            if kind == "AR":
                Vs = [int(V) for V in sorted(ar[key]) if ("AR", key, int(V)) not in self._res]
                if key <= 1:
                    for V in Vs:
                        self.allreduce(key, V)
                elif Vs:
                    # one lazy candidate stream evaluated for every payload of this structure
                    pos = [V for V in Vs if V > 0]
                    res = self._alts_multi(self._ar_stream(key), pos, "AR", f"AR p={key}") if pos else {}
                    for V in Vs:
                        self._res[("AR", key, V)] = res.get(V) or [_zero_alt("AR", self.tp.N, "local")]
                self._ar.pop(key, None)
            elif kind == "AG":
                p_, q_ = key
                Vs = [int(V) for V in sorted(ag[key]) if ("AG", p_, q_, int(V)) not in self._res]
                if p_ <= 1 and q_ <= 1:
                    for V in Vs:
                        self.allgather(p_, q_, V)
                elif Vs:
                    pos = [V for V in Vs if V > 0]
                    res = self._alts_multi(self._ag_stream(p_, q_), pos, "AG", f"AG p={p_} q={q_}") if pos else {}
                    for V in Vs:
                        self._res[("AG", p_, q_, V)] = res.get(V) or [_zero_alt("AG", self.tp.N, "local")]
                self._ag.pop(key, None)
            else:
                s_, d_, b_ = a2a[key]
                self.alltoall(key, s_, d_, b_)
            dt = _t.perf_counter() - t0
            st["seconds"][kind] += dt
            done += 1
            if dt > 5.0:
                import gc
                gc.collect()
            if log is not None and (dt > 5.0 or done % 500 == 0):
                try:
                    rss = int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30
                except Exception:
                    rss = float("nan")
                log(f"prefetch {done}/{len(items)} {kind} {pq} {dt:.1f}s rss={rss:.2f}GiB")
        return st

    # ------------------------------------------------------------------ evaluation
    def _allowed(self, ops):
        """Declared restrictions for the collective comparison (docs/experiments.md):
        one-stage: WRHT m=p and one-stage OpTree/direct candidates only;
        deepest:   WRHT m=2 and the deepest mixed-radix OpTree of each family."""
        if self.family == "all" or not ops:
            return ops
        def depth(op):
            return op.meta.get("depth", 0) if isinstance(op.meta, dict) else 0
        if ops[0].op == "AR":
            ms = [op.meta.get("m", 0) for op in ops]
            target = max(ms) if self.family == "one-stage" else min(ms)
            return [op for op in ops if op.meta.get("m") == target]
        out = []
        fams = {}
        for op in ops:
            fams.setdefault(op.candidate.split("|")[0], []).append(op)
        for fam, lst in fams.items():
            if fam in ("C-C", "E-C"):
                if self.family == "one-stage":
                    out += lst
                continue
            ds = [op.meta.get("depth", op.meta.get("radices") and len(op.meta["radices"]) or 0) for op in lst]
            key = [int(str(op.candidate).split("k=")[-1].split("|")[0]) if "k=" in op.candidate else 0 for op in lst]
            tgt = min(key) if self.family == "one-stage" else max(key)
            out += [op for op, k in zip(lst, key) if k == tgt]
        return out

    def _alts(self, ops, payload, opname, logkey=None):
        return self._alts_list(ops, [payload], opname, logkey)[0]

    def _alts_multi(self, ops, payloads, opname, logkey=None):
        res = self._alts_list(ops, list(payloads), opname, logkey)
        return {V: r for V, r in zip(payloads, res)}

    def family_log(self) -> dict:
        """Candidate identifiers retained by the family restriction, per structure."""
        return {k: sorted(v) for k, v in getattr(self, "_flog", {}).items()}

    def _alts_list(self, ops, payloads, opname, logkey=None):
        """Evaluate a (possibly lazy) candidate stream for several payloads,
        holding one candidate structure at a time."""
        N = self.tp.N
        if self.family != "all":
            ops = self._allowed(list(ops))
            if logkey is not None:
                if not hasattr(self, "_flog"):
                    self._flog = {}
                self._flog[logkey] = [op.candidate for op in ops]
        rows = [[] for _ in payloads]
        for op in ops:
            for k, V in enumerate(payloads):
                ev = evaluate(op, V, self.tp, self.cp.beta_red)
                led = dict(relay=ev.relay_peak.copy(), retained=ev.recv_buffers.copy(),
                           accum=ev.accumulators.copy(), assembled=ev.gathered.copy())
                rows[k].append((_tie_key(op, ev), Alt(opname, op.candidate, ev.transport, ev.reduction, led,
                                                      ev.steps, ev.rounds, ev.setups)))
            del op
        if payloads and not rows[0]:
            # a network operation must have at least one complete candidate; an empty
            # (e.g. over-restricted) set is an error, never a zero-cost operation
            raise ValueError(f"no candidate schedule for {logkey or opname} (family={self.family})")
        out = []
        for r in rows:
            r.sort(key=lambda x: x[0])
            out.append(pareto([a for _, a in r]))
        return out

    def _ar_stream(self, p):
        return self._ar[p] if p in self._ar else F.iter_wrht_ops(list(range(1, p + 1)), self.tp)

    def _ag_stream(self, p, q):
        return self._ag[(p, q)] if (p, q) in self._ag else F.iter_allgather_ops(p, q, self.tp)

    def allreduce(self, p: int, V: int) -> list:
        """T_2(p,p; 8V): WRHT candidates (all m, both top modes)."""
        key = ("AR", p, int(V))
        if key not in self._res:
            if p <= 1 or V <= 0:
                self._res[key] = [_zero_alt("AR", self.tp.N, "local")]
            else:
                ops = self.ar_ops(p) if self.keep_structures else self._ar_stream(p)
                self._res[key] = self._alts(ops, int(V), "AR", f"AR p={p}")
        return self._res[key]

    def allgather(self, p: int, q: int, V: int) -> list:
        """T_3(p,q; 8V): every complete equal/unequal-group candidate."""
        key = ("AG", p, q, int(V))
        if key not in self._res:
            if (p <= 1 and q <= 1) or V <= 0:
                self._res[key] = [_zero_alt("AG", self.tp.N, "local")]
            else:
                ops = self.ag_ops(p, q) if self.keep_structures else self._ag_stream(p, q)
                self._res[key] = self._alts(ops, int(V), "AG", f"AG p={p} q={q}")
        return self._res[key]

    def allreduce_list(self, nodes: tuple, V: int) -> list:
        """T_2 over an explicit ordered physical node list (e.g. GPipe stage replicas)."""
        nodes = tuple(int(x) for x in nodes)
        key = ("ARL", nodes, int(V))
        if key not in self._res:
            if len(nodes) <= 1 or V <= 0:
                self._res[key] = [_zero_alt("AR", self.tp.N, "local")]
            else:
                self._res[key] = self._alts(F.iter_wrht_ops(list(nodes), self.tp), int(V), "AR")
        return self._res[key]

    def alltoall(self, key, src, dst, nbytes) -> list:
        """T_1 with the adapted OSM schedule (one alternative)."""
        rk = ("A2A", key)
        if rk not in self._res:
            src = np.asarray(src, np.int64)
            if src.size == 0:
                self._res[rk] = [_zero_alt("A2A", self.tp.N, "empty")]
            else:
                op = F.osm(src, dst, self.tp)
                self._res[rk] = self._alts([op], np.asarray(nbytes, np.int64), "A2A")
        return self._res[rk]


__all__ = ["Alt", "CollectiveService", "pareto", "CATS"]
