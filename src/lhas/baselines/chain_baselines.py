"""Chain-workload baselines evaluated with the same cost model, collectives and ledger.

Every baseline is a restriction of the LHAS configuration space evaluated by the
same exact memory-aware chain search (lhas.planner.chain_dp), so schedule
alternatives are chosen under the same constraints. Every result carries a status
(lhas.baselines.common): feasible, infeasible (proven) or not_established.

* single_node      all layers (DP, 1); a memory-infeasible value is a time-only reference.
* pure_dp(p)       all layers (DP, p) with one common count p (p | B).
* pure_dp_best     best common DP count over all p | B, p <= N; unresolved counts make
                   the result an incumbent (D11).
* owt              adapted OWT (D12): CONV DP(N), FC MP with the largest divisor <= N;
                   one fixed plan, no repair heuristic (lhas.baselines.common.owt_configs).
* flexflow_mcmc    FlexFlow-derived Metropolis search (D14) over the same per-layer
                   decisions and evaluator, from multiple initial strategies.
* ablations        strategy-only, DP-only, MP-only and greedy layer-wise planners.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..model.chain import ChainModel, DP, MP
from ..planner.chain_dp import plan_chain, PlanResult, LabelLimitExceeded
from . import common as C

# Cap for exact alternative selection inside fixed-configuration baselines. When
# exceeded the evaluation is reported as not completed (never silently dropped).
BASELINE_MAX_LABELS = 50_000


class _Restricted:
    """View of a ChainModel whose configuration sets are restricted."""

    def __init__(self, model: ChainModel, allowed):
        self._m = model
        self._allowed = allowed

    def __getattr__(self, k):
        return getattr(self._m, k)

    def configs(self, i, memory_filter=True):
        base = self._m.configs(i, memory_filter)
        return [c for c in base if c in self._allowed[i]]


def evaluate_fixed(model: ChainModel, seq, max_labels: int | None = BASELINE_MAX_LABELS) -> PlanResult:
    """Exact cost of a fixed configuration sequence, choosing schedule
    alternatives optimally under the memory constraint. In profiled mode a sequence with
    an unmeasured (layer, configuration) is not evaluated and is reported `unavailable`
    with the missing keys (T6); this says nothing about memory feasibility."""
    missing = model.missing_profile(seq)
    if missing:
        return PlanResult(False, float("inf"), [], [], np.zeros(model.N), {},
                          dict(reason="missing compute profile", missing=missing, unavailable=True))
    try:
        return plan_chain(_Restricted(model, [{c} for c in seq]), max_labels=max_labels)
    except LabelLimitExceeded as e:
        inc = getattr(e, "incumbent", None)
        return PlanResult(False, float("inf"), [], [], np.zeros(model.N), {},
                          dict(reason="label limit exceeded; evaluation not completed", detail=str(e),
                               not_completed=True, lower_bound=getattr(e, "lower_bound", None),
                               feasible_upper_bound=inc if inc is not None and np.isfinite(inc) else None))


def single_node(model: ChainModel) -> dict:
    seq = [(DP, 1)] * len(model.L)
    missing = model.missing_profile(seq)
    if missing:
        return dict(name="single-node", status=C.UNAVAILABLE, feasible=False, cost=float("inf"), time_only=None,
                    reason="missing compute profile", missing=missing, configs=seq)
    r = evaluate_fixed(model, seq)
    t_only = sum(model.tcp(i, (DP, 1)) for i in range(len(model.L)))
    mem = sum(model.m(i, (DP, 1)) for i in range(len(model.L)))
    return dict(name="single-node", status=r.status, feasible=r.feasible,
                cost=r.cost if r.feasible else float("inf"),
                time_only=t_only, memory_node1=float(mem[0]), configs=seq)


def pure_dp(model: ChainModel, p: int) -> PlanResult:
    return evaluate_fixed(model, [(DP, p)] * len(model.L))


def pure_dp_best(model: ChainModel):
    """Best common DP count (D11). Returns (summary dict, best PlanResult or None).
    Unresolved counts are kept: the best completed result is then an incumbent."""
    res, plans = {}, {}
    bounds, missing = {}, {}
    for p in [p for p in range(1, model.N + 1) if model.st.B % p == 0]:
        r = pure_dp(model, p)
        res[p] = (r.status, r.cost)
        plans[p] = r
        if r.status == C.NOT_ESTABLISHED:
            bounds[p] = (r.stats.get("lower_bound"), r.stats.get("feasible_upper_bound"))
        if r.status == C.UNAVAILABLE:
            missing[p] = r.stats.get("missing")
    status, bp, bc, unresolved = C.summarize_counts(res, {p: b[0] for p, b in bounds.items()})
    per = {}
    for k, v in res.items():
        e = dict(status=v[0], cost=v[1])
        if k in bounds:
            e.update(lower_bound=bounds[k][0], feasible_upper_bound=bounds[k][1])
        if k in missing:
            e.update(reason="missing compute profile", missing=missing[k])
        per[int(k)] = e
    return dict(status=status, p=bp, cost=bc, unresolved_counts=unresolved,
                unavailable_counts=sorted(missing),
                certification=C.certification(status, unresolved, missing),
                candidate_domain=C.candidate_domain(missing),
                per_count=per), \
        (plans[bp] if bp is not None else None)


def owt(model: ChainModel):
    """Adapted OWT (D12): fixed largest-divisor rule, no repair heuristic."""
    seq = C.owt_configs(model.L, model.N, model.st.B)
    if seq is None:
        return None, None
    return seq, evaluate_fixed(model, seq)


MCMC_MAX_LABELS = 50_000
MCMC_RETRY_LABELS = 200_000
MCMC_RETRY_LABEL_ENTRIES = 51_200_000     # memory bound of the retry: labels x N reservation entries


def retry_label_cap(N: int) -> int:
    """Retry cap min(200 000, 51.2e6 / N): identical for N <= 256; bounded label-matrix
    memory (about 0.4 GB of float64 reservation entries) at N = 512 and 1024."""
    return min(MCMC_RETRY_LABELS, MCMC_RETRY_LABEL_ENTRIES // N)


def chain_evaluator(model: ChainModel, max_labels=MCMC_MAX_LABELS, retry_labels=None):
    """(status, cost) evaluator for the Metropolis search. An evaluation stopped at the
    first label cap is retried once with a larger cap before being reported unresolved."""
    if retry_labels is None:
        retry_labels = retry_label_cap(model.N)

    def ev(seq):
        r = evaluate_fixed(model, seq, max_labels=max_labels)
        if r.status == C.NOT_ESTABLISHED and retry_labels:
            r = evaluate_fixed(model, seq, max_labels=retry_labels)
        return r.status, (r.cost if r.feasible else math.inf)
    return ev


def flexflow_mcmc(model: ChainModel, iterations: int = 1000, seed: int = 0, starts=None):
    """FlexFlow-derived Metropolis search (D14) from multiple initial strategies:
    best-common-count DP, adapted OWT, DP(N) and a feasible random plan."""
    A = [model.configs(i) for i in range(len(model.L))]
    ev = chain_evaluator(model)
    rng = np.random.default_rng(10_000 + seed)
    # the random feasible start is searched inside the search's accounting (F4)
    return C.metropolis(ev, A, list(starts or []), iterations, seed, random_start=dict(rng=rng, tries=50))


# ----------------------------------------------------------------------------
# planner ablations (reviewer request: impact of the DP algorithm and of the
# DP/MP and node-count freedoms); all use the same exact evaluator
# ----------------------------------------------------------------------------
def lhas_restricted(model: ChainModel, allowed) -> PlanResult:
    """Exact planner over a restricted configuration space. allowed(i, c) -> bool."""
    sets = [set(c for c in model.configs(i) if allowed(i, c)) for i in range(len(model.L))]
    return plan_chain(_Restricted(model, sets))


def strategy_only(model: ChainModel) -> PlanResult:
    """DP/MP chosen per layer, node count fixed: DP uses N, MP the largest divisor <= N."""
    N = model.N
    def ok(i, c):
        if c[0] == DP:
            return c[1] == N
        return c[1] == max(p for p in range(1, N + 1) if model.L[i].n_out % p == 0)
    return lhas_restricted(model, ok)


def dp_only_counts(model: ChainModel) -> PlanResult:
    """Per-layer node counts with DP only (no MP)."""
    return lhas_restricted(model, lambda i, c: c[0] == DP)


def mp_only_counts(model: ChainModel) -> PlanResult:
    """Per-layer node counts with MP only (no DP)."""
    return lhas_restricted(model, lambda i, c: c[0] == MP)


def greedy_layerwise(model: ChainModel):
    """Choose each layer's configuration from its own charge only: computation and
    intra-layer collectives, with the MP output All-gather to its own prefix
    (ChainModel.local_charge). Boundary exchanges are ignored during selection and
    charged when the resulting sequence is evaluated exactly."""
    seq = []
    for i in range(len(model.L)):
        best = None
        for c in model.configs(i):
            g = min(a.G(model.tcp(i, c), model.st.alpha) for a in model.local_charge(i, c))
            if best is None or g < best[0]:
                best = (g, c)
        seq.append(best[1])
    return seq, evaluate_fixed(model, seq)


