"""Pure-Python reference implementations of the written routing and packing rules.

Written from the manuscript text (sec:optical_model, sec:transport,
sec:group_comm) without reference to the numba kernels. Slow by design.
"""
from __future__ import annotations


def circ(u, v, N):
    return min((u - v) % N, (v - u) % N)


def route_rule(u, v, N, L):
    """Shorter circular arc, clockwise on ties; relays every L segments with a
    possibly shorter final hop. Returns (direction, node list)."""
    cw, ccw = (v - u) % N, (u - v) % N
    d = 1 if cw <= ccw else -1
    D = min(cw, ccw)
    nodes, x = [u], u
    while D > 0:
        h = min(L, D)
        x = (x - 1 + d * h) % N + 1
        nodes.append(x)
        D -= h
    return d, nodes


class _RoundState:
    def __init__(self, P):
        self.P = P
        self.seg = set()      # ((segment start 0-based, dir), label)
        self.tx = set()       # (node, label)
        self.rx = set()
        self.txc = {}
        self.rxc = {}
        self.peers = {}

    def try_hop(self, a, b, d, ns):
        """Return the lowest feasible label or None (no reservation)."""
        P = self.P
        N = P["N"]
        if self.txc.get(a, 0) >= P["w_tx"] or self.rxc.get(b, 0) >= P["w_rx"]:
            return None
        if P["d_peer"] > 0:
            pa, pb = self.peers.get(a, set()), self.peers.get(b, set())
            if (b not in pa and len(pa) >= P["d_peer"]) or (a not in pb and len(pb) >= P["d_peer"]):
                return None
        segs = [((a - 1 + d * t) % N, d) for t in range(ns)]
        for lab in range(1, P["lam"] + 1):
            if (a, lab) in self.tx or (b, lab) in self.rx:
                continue
            if any((s, lab) in self.seg for s in segs):
                continue
            return lab
        return None

    def reserve(self, a, b, d, ns, lab):
        N = self.P["N"]
        for t in range(ns):
            self.seg.add((((a - 1 + d * t) % N, d), lab))
        self.tx.add((a, lab)); self.rx.add((b, lab))
        self.txc[a] = self.txc.get(a, 0) + 1
        self.rxc[b] = self.rxc.get(b, 0) + 1
        self.peers.setdefault(a, set()).add(b)
        self.peers.setdefault(b, set()).add(a)

    def snapshot(self):
        return (set(self.seg), set(self.tx), set(self.rx), dict(self.txc), dict(self.rxc),
                {k: set(v) for k, v in self.peers.items()})

    def restore(self, s):
        self.seg, self.tx, self.rx, self.txc, self.rxc, self.peers = s


def _hop_list(u, v, P):
    N, L = P["N"], P["reach"]
    d, nodes = route_rule(u, v, N, L)
    out = []
    for k in range(len(nodes) - 1):
        a, b = nodes[k], nodes[k + 1]
        ns = ((b - a) * d) % N
        out.append((a, b, d, ns))
    return out


def reference_first_fit(routes, P):
    """routes: list of (src, dst, item). Returns sorted list of
    (round, src, dst, item, labels tuple) following the written rule."""
    N = P["N"]
    info = []
    for (u, v, it) in routes:
        hops = _hop_list(u, v, P)
        D = sum(h[3] for h in hops)
        info.append(((-len(hops), -D, u, v, it), u, v, it, hops))
    info.sort(key=lambda x: x[0])
    pending = info
    out = []
    rnd = 0
    while pending:
        st = _RoundState(P)
        nxt = []
        placed = 0
        for key, u, v, it, hops in pending:
            snap = st.snapshot()
            labs = []
            ok = True
            for (a, b, d, ns) in hops:
                lab = st.try_hop(a, b, d, ns)
                if lab is None:
                    ok = False
                    break
                st.reserve(a, b, d, ns, lab)
                labs.append(lab)
            if ok:
                out.append((rnd, u, v, it, tuple(labs)))
                placed += 1
            else:
                st.restore(snap)
                nxt.append((key, u, v, it, hops))
        if placed == 0:
            raise AssertionError("reference first-fit made no progress")
        pending = nxt
        rnd += 1
    return sorted(out)


def reference_dissemination_round(targets, shards, need, hold, P):
    """One E-A dissemination round as written in sec:group_comm.

    Targets in the given order; for each target, missing shards in ascending
    identifier order; direct senders (within reach) holding the shard at round
    start, by increasing circular distance then identifier; first feasible
    lowest-label assignment. Returns list of (src, dst, shard, label)."""
    N, L = P["N"], P["reach"]
    st = _RoundState(P)
    out = []
    for v in targets:
        cands = sorted([u for u in range(1, N + 1) if u != v and circ(u, v, N) <= L],
                       key=lambda u: (circ(u, v, N), u))
        for s in sorted(need.get(v, ())):
            if st.rxc.get(v, 0) >= P["w_rx"]:
                break
            for u in cands:
                if s not in hold.get(u, ()):
                    continue
                d, nodes = route_rule(u, v, N, L)
                if len(nodes) != 2:
                    continue
                ns = circ(u, v, N)
                lab = st.try_hop(u, v, d, ns)
                if lab is None:
                    continue
                st.reserve(u, v, d, ns, lab)
                out.append((u, v, s, lab))
                break
    return out
