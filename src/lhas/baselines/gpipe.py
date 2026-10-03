"""Optical-adapted GPipe baseline (contract D13, final). Chain workloads only.

Search space (all evaluated on the same transport, compute and memory model):
* S contiguous stages, 1 <= S <= number of weighted layers; S = 1 is a degenerate
  data-parallel reference (one fused gradient All-reduce) and is labeled as such;
* R data-replicated pipelines with R | B and active nodes P = R*S <= N; replica r
  occupies physical nodes r*S+1 .. r*S+S; stage s of all replicas forms the ordered
  list [s+1, S+s+1, ...] for its gradient All-reduce;
* M microbatches, M in {1, 2, 4, 8, 16, 32}, with B/(R*M) an integer;
* rematerialization off and on (on: the stage forward is recomputed in the backward
  pass and only stage inputs are stashed for all microbatches).
Stage partition: contiguous partition minimizing the largest per-example stage
compute (exact linear-partition dynamic program); this is a restricted partition
choice, not a globally optimal pipeline partition.

Timing: a dependency/resource schedule simulation of the synchronous GPipe
schedule (all forwards, then all backwards in reverse microbatch order). Each stage
processes one microbatch at a time; each stage boundary carries one microbatch at a
time in each direction; transfers overlap with computation. Forward and backward
boundary circuits of all replicas are configured once per pass (one setup each) and
are checked to fit simultaneously in one first-fit round; if they did not fit, every
microbatch transfer would be charged as a separately configured operation. The
per-stage gradient All-reduces follow the backward pass and are charged additively,
as in the LHAS objective. Local optimizer updates are excluded, as for all methods.

Memory per node: stage parameters, gradients and momentum (declared factor);
activations (without rematerialization: stored layer inputs and outputs of all M
microbatches plus one microbatch's gradients; with rematerialization: stage inputs
of all M microbatches plus one microbatch's activations and gradients); the
All-reduce ledger of the selected alternative; the runtime reserve through the
budget. All-reduce alternatives are chosen jointly across stages under the budget
(exact search over nondominated alternatives), not only the fastest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..collectives import families as F
from ..model.chain import ChainModel
from ..model.layercost import param_numel
from ..transport.schedule import evaluate

MICRO = (1, 2, 4, 8, 16, 32)


def _partition(times, S):
    n = len(times)
    pre = np.concatenate([[0.0], np.cumsum(times)])
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(S + 1)]
    arg = [[0] * (n + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        for j in range(s, n + 1):
            for i in range(s - 1, j):
                v = max(dp[s - 1][i], pre[j] - pre[i])
                if v < dp[s][j] - 1e-18:
                    dp[s][j], arg[s][j] = v, i
    cuts, j = [], n
    for s in range(S, 0, -1):
        i = arg[s][j]
        cuts.append((i, j))
        j = i
    return list(reversed(cuts))


def simulate_pipeline(tf, tb, cf, cb, M):
    """Synchronous GPipe schedule. tf, tb: per-stage compute per microbatch;
    cf[s], cb[s]: transfer time on boundary s -> s+1 (forward) and s+1 -> s
    (backward) per microbatch. Returns (forward end, makespan): both are absolute
    times from the start of the iteration; the backward pass occupies
    makespan - forward end."""
    S = len(tf)
    endF = [[0.0] * (M + 1) for _ in range(S)]
    link = [0.0] * S
    for m in range(1, M + 1):
        for s in range(S):
            ready = 0.0
            if s > 0:
                start_x = max(endF[s - 1][m], link[s - 1])
                link[s - 1] = start_x + cf[s - 1]
                ready = link[s - 1]
            start = max(endF[s][m - 1], ready)
            endF[s][m] = start + tf[s]
    t_fwd_end = max(endF[s][M] for s in range(S))
    endB = [[0.0] * (M + 2) for _ in range(S)]
    linkb = [0.0] * S
    order = list(range(M, 0, -1))
    prevB = [endF[s][M] for s in range(S)]
    for m in order:
        for s in range(S - 1, -1, -1):
            ready = endF[S - 1][M] if s == S - 1 else 0.0
            if s < S - 1:
                start_x = max(endB[s + 1][m], linkb[s])
                linkb[s] = start_x + cb[s]
                ready = linkb[s]
            start = max(prevB[s], ready, endF[s][M])
            endB[s][m] = start + tb[s]
            prevB[s] = endB[s][m]
    return t_fwd_end, max(prevB)            # (forward end, makespan of the whole iteration)


@dataclass
class GPipeResult:
    feasible: bool
    cost: float
    S: int
    R: int
    P: int
    M: int
    remat: bool
    stages: list
    breakdown: dict
    memory_max: float
    degenerate_dp: bool
    status: str = "feasible"
    stats: dict = field(default_factory=dict)


AR_SELECT_NODE_LIMIT = 200_000


def _ar_select(stage_alts, static_mem, budget, node_limit=AR_SELECT_NODE_LIMIT):
    """Choose one All-reduce alternative per stage minimizing the summed latency
    subject to static_mem + sum of ledgers <= budget on every node. Exact depth-first
    branch-and-bound: alternatives in ascending latency; reservations only grow, so a
    partial violation prunes; the bound adds, for every remaining stage, the fastest
    alternative that still fits individually on top of the current reservations (and
    prunes when a stage has none). Returns (latency, memory, candidates, complete) or
    None when no combination fits; complete=False if the node limit stopped the search
    before optimality was established (the incumbent is then returned and flagged)."""
    alts = [sorted(al, key=lambda x: x[0].latency) for al in stage_alts]
    n = len(alts)
    lat = [np.array([a.latency for a, _ in al]) for al in alts]
    vecs = [np.stack([v for _, v in al]) for al in alts]
    best = [math.inf, None, None]
    visited = [0]
    stopped = [False]

    def bound(k, mem):
        tot = 0.0
        for j in range(k, n):
            ok = np.all(vecs[j] + mem <= budget + 1e-6, axis=1)
            if not ok.any():
                return math.inf
            tot += lat[j][int(np.argmax(ok))]         # sorted by latency: first feasible is fastest
        return tot

    def rec(k, t, mem, ch):
        if k == n:
            if t < best[0]:
                best[0], best[1], best[2] = t, mem, ch
            return
        for idx, (a, vec) in enumerate(alts[k]):
            if t + a.latency >= best[0] - 1e-15:
                break
            visited[0] += 1
            if visited[0] > node_limit:
                stopped[0] = True
                return
            m2 = mem + vec
            if not np.all(m2 <= budget + 1e-6):
                continue
            if t + a.latency + bound(k + 1, m2) >= best[0] - 1e-15:
                continue
            rec(k + 1, t + a.latency, m2, ch + [a.cand])
            if stopped[0]:
                return

    if bound(0, static_mem) < math.inf:
        rec(0, 0.0, static_mem.copy(), [])
    if best[1] is None:
        return None if not stopped[0] else (math.inf, None, None, False)
    return best[0], best[1], best[2], not stopped[0]


def gpipe(model: ChainModel, micro=MICRO, remats=(False, True), stage_counts=None):
    tp, cp, st = model.tp, model.cp, model.st
    svc = model.svc
    N, B, psi, beta = model.N, st.B, model.psi, cp.beta_red
    Ls = model.L
    n = len(Ls)
    rate = cp.utilization * cp.peak_flops
    f1 = [L.mult_flops_per_example() / rate for L in Ls]
    b1 = [(2 if model.needs_input_grad(L) else 1) * L.mult_flops_per_example() / rate for L in Ls]
    k_opt = st.optimizer_state_factor
    budget = model.budget
    cands = []
    for S in (stage_counts or range(1, n + 1)):
        if S > N:
            continue
        stages = _partition([a + b for a, b in zip(f1, b1)], S)
        for R in [r for r in range(1, N // S + 1) if B % r == 0]:
            for M in micro:
                if B % (R * M):
                    continue
                b = B // (R * M)
                for remat in remats:
                    tf_ = [sum(f1[i] for i in range(a, z)) * b for a, z in stages]
                    tb_ = [sum(b1[i] for i in range(a, z)) * b + (tf_[k] if remat else 0.0)
                           for k, (a, z) in enumerate(stages)]
                    # valid lower bound on the simulated makespan (transfers, setup, All-reduce >= 0)
                    lb = sum(tf_) + sum(tb_) + (M - 1) * (max(tf_) + max(tb_))
                    cands.append((lb, S, R, M, remat, stages))
    cands.sort(key=lambda x: x[0])
    best, evaluated, pruned, unresolved = None, 0, 0, 0
    circuit_checks = {}
    for (lb, S, R, M, remat, stages) in cands:
        if best is not None and lb >= best.cost:
            pruned += 1
            continue
        evaluated += 1
        b = B // (R * M)
        tf = [sum(f1[i] for i in range(a, z)) * b for a, z in stages]
        tb = [sum(b1[i] for i in range(a, z)) * b + (tf[k] if remat else 0.0) for k, (a, z) in enumerate(stages)]
        # boundary transfers: simultaneous configuration check and per-microbatch times
        cf, cb, setup = [], [], 0.0
        if S > 1:
            key = (S, R)
            if key not in circuit_checks:
                srcF = [r * S + k + 1 for k in range(S - 1) for r in range(R)]
                dstF = [r * S + k + 2 for k in range(S - 1) for r in range(R)]
                opF = F.osm(np.array(srcF), np.array(dstF), tp)
                opB = F.osm(np.array(dstF), np.array(srcF), tp)
                circuit_checks[key] = (opF.nrounds == 1 and opB.nrounds == 1)
            simultaneous = circuit_checks[key]
            for k, (a, z) in enumerate(stages[:-1]):
                V = psi * Ls[z - 1].u_numel * b
                pairs = [(r * S + k + 1, r * S + k + 2) for r in range(R)]
                eF = evaluate(F.osm(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]), tp), int(V), tp, beta)
                eB = evaluate(F.osm(np.array([q for _, q in pairs]), np.array([p for p, _ in pairs]), tp), int(V), tp, beta)
                if simultaneous:
                    cf.append(eF.transport - eF.setups * tp.t_setup)
                    cb.append(eB.transport - eB.setups * tp.t_setup)
                else:
                    cf.append(eF.transport); cb.append(eB.transport)
            if simultaneous:
                setup = 2 * tp.t_setup
        else:
            simultaneous = True
        t_fwd, t_end = simulate_pipeline(tf, tb, cf, cb, M)
        # static memory per node
        mem = np.zeros(N)
        for k, (a, z) in enumerate(stages):
            w = sum(param_numel(Ls[i], st.norm_policy) for i in range(a, z)) * psi
            act_mb = sum(Ls[i].x_numel + Ls[i].y_numel for i in range(a, z)) * b * psi
            if remat:
                act = Ls[a].x_numel * b * psi * M + 2 * act_mb
            else:
                act = act_mb * M + act_mb
            per_node = (2 + k_opt) * w + act + st.workspace_bytes
            for r in range(R):
                mem[r * S + k] += per_node
        if np.any(mem > budget + 1e-6):
            continue
        # gradient All-reduce per stage, chosen jointly under the budget
        stage_alts = []
        for k, (a, z) in enumerate(stages):
            wbytes = psi * sum(param_numel(Ls[i], st.norm_policy) for i in range(a, z))
            if R > 1:
                base = tuple(r * S + 1 for r in range(R))
                alts = svc.allreduce_list(base, int(wbytes))
                # stage k's list is the base list translated by k nodes (ring automorphism)
                stage_alts.append([(al, np.roll(al.mem(), k)) for al in alts])
            else:
                stage_alts.append([(_Zero(), np.zeros(N))])
        sel = _ar_select(stage_alts, mem, budget)
        if sel is None:
            continue
        t_ar, mem_tot, chosen, complete = sel
        if not complete:
            unresolved += 1
            if mem_tot is None:
                continue
        total = t_end + setup + t_ar
        if best is None or total < best.cost:
            best = GPipeResult(True, total, S, R, R * S, M, remat, stages,
                               dict(forward=t_fwd, backward=t_end - t_fwd, setup=setup, allreduce=t_ar,
                                    boundary_circuits_simultaneous=simultaneous, allreduce_candidates=chosen),
                               float(mem_tot.max()), S == 1)
    stats = dict(candidates=len(cands), evaluated=evaluated, pruned_by_bound=pruned,
                 allreduce_selection_unresolved=unresolved)
    if best is not None:
        best.stats = stats
        if unresolved:
            best.status = "incumbent"
    return best, stats


class _Zero:
    latency = 0.0
    cand = "local"


__all__ = ["gpipe", "GPipeResult", "simulate_pipeline", "MICRO"]
