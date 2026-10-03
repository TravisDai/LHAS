"""Independent schedule verifier (implementation-consistency check).

This package imports nothing from `lhas`. It consumes a plain, JSON-serializable
schedule record (phases -> rounds -> routes with explicit hop node sequences,
labels and, optionally, explicit chunk transmissions) together with a semantic
specification and a parameter dictionary. It re-derives every property with
deliberately simple code:

  R1  label range, per-hop reach, hop chaining, route endpoints;
  R2  routing rule: shorter arc, clockwise on ties, fixed-stride relays;
  R3  per-round directed-segment/label conflicts, per-label Tx/Rx uniqueness
      shared across directions, aggregate Tx/Rx budgets, peer-union cap;
  R4  deterministic first-fit conformance (optional; re-packs the phase with an
      independent pure-Python implementation of the written rule);
  R5  chunking (Eq. chunks, actual tail), step rule a+j-1, relay causality,
      one chunk per circuit per step, whole items in direct-only rounds;
  R6  data availability at round start and final ownership / delivery;
  R7  All-reduce contribution multiplicity and declared reduction counts;
  R8  setup flags from the retained configuration of the same operation;
  R9  transport latency (Eq. bg_generic_collective) and local reduction
      (Eq. local_reduction);
  R10 buffers: relay double-buffer reservation covers actual chunk occupancy,
      and claimed reservations are not understated.

Passing this verifier shows agreement between the constructor and the written
model on the checked instances. It is not physical validation of the model.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .packing import reference_first_fit, reference_dissemination_round, route_rule, circ

REL = 1e-9


class VerificationError(AssertionError):
    """Raised with a message naming the violated rule (R1..R10)."""


def _fail(code, msg):
    raise VerificationError(f"[{code}] {msg}")


def _close(a, b):
    return abs(a - b) <= 1e-15 + REL * max(abs(a), abs(b))


def _chunks(V, kappa):
    if kappa <= 0:
        return [V]
    c = -(-V // kappa)
    return [min(kappa, V - (j - 1) * kappa) for j in range(1, c + 1)]


def verify(record: dict, spec: dict, params: dict, check_rule: bool = True,
           check_packing: bool = True) -> dict:
    N = int(params["N"]); lam = int(params["lam"])
    wtx = int(params["w_tx"]); wrx = int(params["w_rx"])
    L = int(params["reach"]) if int(params["reach"]) > 0 else N
    dpeer = int(params["d_peer"]); kappa = int(params["kappa_bytes"])
    rate = float(params["rate_bps"]) * float(params["eps"])
    t_tx, t_rx, t_fwd = float(params["t_tx"]), float(params["t_rx"]), float(params["t_fwd"])
    seg_t = float(params["seg_len_m"]) / float(params["v_g"])
    t_setup = float(params["t_setup"])
    beta = params.get("beta_red")
    P = dict(N=N, lam=lam, w_tx=wtx, w_rx=wrx, reach=L, d_peer=dpeer)

    op = record["op"]
    if op == "AG":
        hold = {int(k): set(v) for k, v in spec["initial"].items()}
        shard_bytes = {int(k): int(v) for k, v in spec["shard_bytes"].items()}
    elif op == "AR":
        parts = [int(u) for u in spec["participants"]]
        val = {u: Counter({u: 1}) for u in parts}
        Vc = int(spec["contribution_bytes"])
    elif op == "A2A":
        required = {(int(u), int(v)): int(b) for (u, v), b in spec["required"].items()}
        delivered = Counter()
    else:
        _fail("R6", f"unknown operation {op}")

    transport = 0.0; reduction = 0.0
    steps_total = setups = nrounds = 0
    retained = None
    relay_occ = defaultdict(float)      # actual peak chunk occupancy
    relay_res = defaultdict(float)      # manuscript reservation 2 min(kappa, V) per route
    gathered = defaultdict(float)
    recv_peak = defaultdict(float)
    acc = defaultdict(float)

    for pi, ph in enumerate(record["phases"]):
        sem = ph.get("semantics", "transfer")
        received = defaultdict(list)
        need = None
        if ph.get("rule") == "dissemination":
            if op != "AG":
                _fail("R4", "dissemination rule outside All-gather")
            if "P" in spec and "Q" in spec:
                exp_t = expected_targets(spec["P"], spec["Q"], N)
                if list(ph["targets"]) != exp_t:
                    _fail("R4", f"dissemination targets {ph['targets']} differ from rule {exp_t}")
            need = {int(t): set(ph["shards"]) - hold.get(int(t), set()) for t in ph["targets"]}
        phase_start_val = {u: Counter(c) for u, c in val.items()} if op == "AR" else None
        rounds = ph["rounds"]
        # ---------- R4: deterministic packing conformance ----------
        if check_packing and rounds:
            rule = ph.get("rule", "first_fit")
            if rule == "first_fit":
                flat = [r for rnd in rounds for r in rnd["routes"]]
                ref = reference_first_fit([(r["src"], r["dst"], r["item"]) for r in flat], P)
                got = []
                for q, rnd in enumerate(rounds):
                    for r in rnd["routes"]:
                        got.append((q, r["src"], r["dst"], r["item"], tuple(h[4] for h in r["hops"])))
                if sorted(got) != sorted(ref):
                    diff = sorted(set(got) ^ set(ref))[:4]
                    _fail("R4", f"phase {ph.get('name')} differs from deterministic first-fit: {diff}")
        for rnd in rounds:
            nrounds += 1
            routes = rnd["routes"]
            if not routes:
                _fail("R5", "empty round")
            if op == "AG":
                start_hold = {k: set(v) for k, v in hold.items()}
            # ---------- R4 (dissemination rule) ----------
            if check_packing and ph.get("rule") == "dissemination" and op == "AG":
                ref = reference_dissemination_round(ph["targets"], ph["shards"], need, start_hold, P)
                got = sorted((r["src"], r["dst"], r["item"], r["hops"][0][4]) for r in routes)
                if got != sorted(ref):
                    _fail("R4", f"dissemination round differs from the written rule: "
                                f"{sorted(set(got) ^ set(ref))[:4]}")
            seg_lab = set(); txl = set(); rxl = set()
            txc = Counter(); rxc = Counter(); peers = defaultdict(set)
            circuits = set()
            pipelined = any(len(r["hops"]) >= 2 for r in routes)
            tx_all = []    # (step, duration, sender, hop_index, chunk, bytes, route_idx)
            for ri, r in enumerate(routes):
                u, v, V, item = int(r["src"]), int(r["dst"]), int(r["bytes"]), r["item"]
                if u == v:
                    _fail("R6", "self-transfer scheduled")
                if V <= 0:
                    _fail("R6", "zero-byte item scheduled")
                hops = [tuple(h) for h in r["hops"]]
                if not hops:
                    _fail("R1", "route without hops")
                # ---------- R1 / R2 ----------
                if check_rule:
                    d_exp, nodes_exp = route_rule(u, v, N, L)
                    exp = [(nodes_exp[k], nodes_exp[k + 1], d_exp, ((nodes_exp[k + 1] - nodes_exp[k]) * d_exp) % N)
                           for k in range(len(nodes_exp) - 1)]
                    got_h = [tuple(h[:4]) for h in hops]
                    if got_h != exp:
                        _fail("R2", f"route {u}->{v} uses hops {got_h}; rule gives {exp}")
                prev = u
                for (a, b, d, ns, lab) in hops:
                    if a != prev:
                        _fail("R1", "hop chain broken")
                    if not (1 <= lab <= lam):
                        _fail("R1", f"label {lab} outside 1..{lam}")
                    if not (1 <= ns <= L):
                        _fail("R1", f"hop of {ns} segments exceeds reach {L}")
                    if d not in (1, -1) or (b - a) % N != (d * ns) % N:
                        _fail("R1", "hop endpoints inconsistent with direction and segment count")
                    # ---------- R3 ----------
                    for t in range(ns):
                        sg = ((a - 1 + d * t) % N, d)
                        if (sg, lab) in seg_lab:
                            _fail("R3", f"directed segment {sg} carries label {lab} twice")
                        seg_lab.add((sg, lab))
                    if (a, lab) in txl:
                        _fail("R3", f"node {a} transmits label {lab} twice")
                    if (b, lab) in rxl:
                        _fail("R3", f"node {b} receives label {lab} twice")
                    txl.add((a, lab)); rxl.add((b, lab))
                    txc[a] += 1; rxc[b] += 1
                    peers[a].add(b); peers[b].add(a)
                    circuits.add((a, b, d, ns, lab))
                    prev = b
                if prev != v:
                    _fail("R1", "route does not end at its destination")
                # ---------- R6: availability / item size ----------
                if op == "AG":
                    if item not in start_hold.get(u, set()):
                        _fail("R6", f"node {u} sends shard {item} that it does not hold at round start")
                    if V != shard_bytes[item]:
                        _fail("R6", f"shard {item} sent with {V} bytes, expected {shard_bytes[item]}")
                elif op == "AR":
                    if u not in phase_start_val:
                        _fail("R6", f"non-participant {u} sends All-reduce content")
                    if V != Vc:
                        _fail("R6", "All-reduce item size differs from the contribution size")
                # ---------- R5: chunks and steps ----------
                sizes = _chunks(V, kappa) if pipelined else [V]
                txs = r.get("tx")
                if txs is None:
                    txs = [(ai, j, ai + j - 1, sz) for ai in range(1, len(hops) + 1)
                           for j, sz in enumerate(sizes, start=1)]
                per_hop = defaultdict(dict)
                for (ai, j, st, sz) in txs:
                    if not (1 <= ai <= len(hops)):
                        _fail("R5", "transmission on a nonexistent hop")
                    if j in per_hop[ai]:
                        _fail("R5", "chunk transmitted twice on one hop")
                    per_hop[ai][j] = (st, sz)
                for ai in range(1, len(hops) + 1):
                    got_sizes = [per_hop[ai].get(j, (None, None))[1] for j in range(1, len(sizes) + 1)]
                    if len(per_hop[ai]) != len(sizes) or got_sizes != sizes:
                        if not pipelined:
                            _fail("R5", "direct-only round must send each item whole in one step")
                        _fail("R5", f"hop {ai} chunks {got_sizes} differ from Eq. chunks {sizes}")
                    steps_on_hop = [per_hop[ai][j][0] for j in range(1, len(sizes) + 1)]
                    if len(set(steps_on_hop)) != len(steps_on_hop):
                        _fail("R5", "one circuit carries two chunks in the same step")
                for ai in range(2, len(hops) + 1):
                    for j in range(1, len(sizes) + 1):
                        if not per_hop[ai][j][0] > per_hop[ai - 1][j][0]:
                            _fail("R5", "relay forwards a chunk before receiving it completely")
                for ai in range(1, len(hops) + 1):
                    for j in range(1, len(sizes) + 1):
                        st, sz = per_hop[ai][j]
                        if st != ai + j - 1:
                            _fail("R5", f"hop {ai} chunk {j} in step {st}, rule gives {ai + j - 1}")
                        a_, b_, d_, ns_, lab_ = hops[ai - 1]
                        dur = 8.0 * sz / rate + t_tx + ns_ * seg_t + t_rx + (t_fwd if ai > 1 else 0.0)
                        tx_all.append((st, dur, a_, ai, j, sz, ri))
            for n_, c_ in txc.items():
                if c_ > wtx:
                    _fail("R3", f"node {n_} uses {c_} transmitters > {wtx}")
            for n_, c_ in rxc.items():
                if c_ > wrx:
                    _fail("R3", f"node {n_} uses {c_} receivers > {wrx}")
            if dpeer > 0:
                for n_, s_ in peers.items():
                    if len(s_) > dpeer:
                        _fail("R3", f"node {n_} has {len(s_)} direct peers > {dpeer}")
            # ---------- R9: step durations ----------
            S = max(t[0] for t in tx_all)
            per_step = defaultdict(float)
            for (st, dur, *_x) in tx_all:
                per_step[st] = max(per_step[st], dur)
            if set(per_step) != set(range(1, S + 1)):
                _fail("R5", "empty step inside a round")
            body = sum(per_step[s] for s in range(1, S + 1))
            # ---------- R8: setup ----------
            if retained is not None and circuits <= retained:
                r_flag = 0
            else:
                r_flag = 1
                retained = set(circuits)
            if "setup" in rnd and int(rnd["setup"]) != r_flag:
                _fail("R8", f"round setup flag {rnd['setup']} but retained configuration gives {r_flag}")
            if "steps" in rnd and int(rnd["steps"]) != S:
                _fail("R9", f"round claims {rnd['steps']} steps; rule gives {S}")
            if "body" in rnd and not _close(float(rnd["body"]), body):
                _fail("R9", f"round claims body {rnd['body']}; recomputed {body}")
            setups += r_flag
            transport += r_flag * t_setup + body
            steps_total += S
            # ---------- R10: relay buffers ----------
            if pipelined:
                occ = defaultdict(float)
                res = defaultdict(float)
                for (st, dur, a_, ai, j, sz, ri) in tx_all:
                    if ai >= 2:
                        # chunk is received during step st-1 and forwarded during st
                        occ[(a_, st - 1)] += sz
                        occ[(a_, st)] += sz
                for r in routes:
                    for h in r["hops"][1:]:
                        V = int(r["bytes"])
                        res[h[0]] += 2 * (min(kappa, V) if kappa > 0 else V)
                peak = defaultdict(float)
                for (node, _st), b_ in occ.items():
                    peak[node] = max(peak[node], b_)
                for node, b_ in peak.items():
                    if b_ > res[node] + 1e-9:
                        _fail("R10", f"relay {node} occupancy {b_} exceeds its reservation {res[node]}")
                    relay_occ[node] = max(relay_occ[node], b_)
                for node, b_ in res.items():
                    relay_res[node] = max(relay_res[node], b_)
            # ---------- R6: deliveries after the round ----------
            for r in routes:
                u, v, item = int(r["src"]), int(r["dst"]), r["item"]
                if op == "AG":
                    if item in hold.setdefault(v, set()):
                        _fail("R6", f"shard {item} delivered twice to node {v}")
                    hold[v].add(item)
                    gathered[v] += int(r["bytes"])
                    if need is not None and v in need:
                        need[v].discard(item)
                elif op == "AR":
                    received[v].append(Counter(phase_start_val[u]))
                else:
                    delivered[(u, v)] += int(r["bytes"])
        # ---------- R7: reductions / replacements ----------
        if op == "AR":
            if sem == "reduce":
                declared = {int(k): int(c) for k, c in ph.get("reduce_counts", {}).items() if int(c) > 0}
                actual = {v: len(c) for v, c in received.items()}
                if declared != actual:
                    _fail("R7", f"declared reduction counts {declared} differ from received {actual}")
                if actual:
                    reduction += max((a + 2) * Vc / beta for a in actual.values())
                    for v, contents in received.items():
                        recv_peak[v] = max(recv_peak[v], len(contents) * Vc)
                        acc[v] = max(acc[v], Vc)
                        tot = Counter(val[v])
                        for c in contents:
                            tot.update(c)
                        if any(x > 1 for x in tot.values()):
                            _fail("R7", f"node {v} counts a contribution more than once")
                        val[v] = tot
            elif sem == "replace":
                for v, contents in received.items():
                    if len(contents) != 1:
                        _fail("R7", "a replace phase must deliver exactly one result per receiver")
                    val[v] = contents[0]
            elif received:
                _fail("R7", "All-reduce phase without declared semantics")
    # ---------- R6 / R7: final conditions ----------
    if op == "AG":
        for node, req in spec["required"].items():
            missing = set(req) - hold.get(int(node), set())
            if missing:
                _fail("R6", f"node {node} misses shards {sorted(missing)[:5]}")
    elif op == "AR":
        target = Counter({u: 1 for u in parts})
        for u in parts:
            if val[u] != target:
                _fail("R7", f"node {u} does not hold every contribution exactly once")
    else:
        for key, b in required.items():
            if delivered[key] != b:
                _fail("R6", f"block {key} delivered {delivered[key]} of {b} bytes")
        extra = set(k for k, b in delivered.items() if b) - set(required)
        if extra:
            _fail("R6", f"unrequested blocks delivered: {sorted(extra)[:5]}")
    out = dict(transport=transport, reduction=reduction, steps=steps_total, setups=setups,
               rounds=nrounds, relay_reservation=dict(relay_res), relay_occupancy=dict(relay_occ),
               gathered=dict(gathered), recv_peak=dict(recv_peak), accumulators=dict(acc))
    # ---------- compare totals and reservations with the constructor ----------
    cl = record.get("claimed")
    if cl is not None:
        if not _close(cl["transport"], transport):
            _fail("R9", f"transport claimed {cl['transport']} != verified {transport}")
        if not _close(cl["reduction"], reduction):
            _fail("R9", f"reduction claimed {cl['reduction']} != verified {reduction}")
        for k in ("steps", "setups", "rounds"):
            if int(cl[k]) != out[k]:
                _fail("R9" if k == "steps" else "R8", f"{k} claimed {cl[k]} != verified {out[k]}")
        for key_c, key_v in (("relay_peak", "relay_reservation"), ("gathered", "gathered"),
                             ("recv_peak", "recv_peak"), ("accumulators", "accumulators")):
            claimed = {int(k): float(x) for k, x in cl[key_c].items()}
            for node, req_b in out[key_v].items():
                if claimed.get(node, 0.0) + 1e-6 < req_b:
                    _fail("R10", f"{key_c} at node {node}: claimed {claimed.get(node, 0.0)} < required {req_b}")
    return out


def expected_targets(P, Q, N):
    """Targets of E-A dissemination: Q minus P ordered by circular distance to P, then id."""
    return sorted([v for v in Q if v not in set(P)], key=lambda v: (min(circ(u, v, N) for u in P), v))
