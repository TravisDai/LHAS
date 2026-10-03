"""Definitions shared by the chain and branch runners (one definition per baseline).

Status vocabulary for every planner/baseline evaluation:
  feasible         a plan satisfying the modeled reservation budget was found; for an
                   exact search it is optimal within the searched space
  infeasible       proven: exact search (or a reservation lower bound) shows no
                   schedule combination of the plan/space fits the budget
  not_established  unresolved: a declared resource limit stopped the evaluation, or the
                   fastest schedule combination failed the budget without a proof that
                   slower combinations also fail
  incumbent        best completed result of a set of evaluations of which at least one
                   is not_established and not excluded by a valid cost lower bound
  unavailable      not evaluated: a required input is missing (profiled mode: a local shape
                   was not measured); says nothing about memory feasibility
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..model.layercost import DP, MP

FEASIBLE, INFEASIBLE, NOT_ESTABLISHED, INCUMBENT = "feasible", "infeasible", "not_established", "incumbent"
UNAVAILABLE = "unavailable"
STATUSES = (FEASIBLE, INFEASIBLE, NOT_ESTABLISHED, UNAVAILABLE)


def owt_configs(layers, N: int, B: int):
    """Adapted OWT (D12): every CONV layer DP(N); every FC layer MP with the largest
    divisor p <= N of its output dimension. No repair heuristic. Returns None if
    N does not divide B (CONV DP(N) undefined)."""
    if B % N:
        return None
    return [(DP, N) if L.kind == "conv" else (MP, max(p for p in range(1, N + 1) if L.n_out % p == 0))
            for L in layers]


def summarize_counts(results: dict, lower_bounds: dict | None = None):
    """results: p -> (status, cost); lower_bounds: p -> lower bound on the cost of an
    unresolved count (e.g. its unconstrained optimum). Returns (status, best_p,
    best_cost, unresolved). The best completed result is certified (FEASIBLE) when every
    unresolved count has a lower bound no smaller than it; otherwise it is an INCUMBENT."""
    lower_bounds = lower_bounds or {}
    feas = {p: c for p, (s, c) in results.items() if s == FEASIBLE}
    unresolved = [p for p, (s, _) in results.items() if s == NOT_ESTABLISHED]
    if feas:
        p = min(feas, key=lambda k: (feas[k], k))
        open_ = [q for q in unresolved if not (lower_bounds.get(q) is not None and lower_bounds[q] >= feas[p])]
        return (INCUMBENT if open_ else FEASIBLE), p, feas[p], unresolved
    if unresolved:
        return NOT_ESTABLISHED, None, math.inf, unresolved
    if results and all(s == UNAVAILABLE for s, _ in results.values()):
        return UNAVAILABLE, None, math.inf, unresolved
    return INFEASIBLE, None, math.inf, unresolved


def certification(status: str, unresolved, unavailable) -> str:
    """How a best-common-count result is certified (F5): 'complete enumeration' (every count
    resolved), 'lower-bound exclusion' (each unresolved count has a lower bound at least the
    best feasible cost) or 'incumbent' (some unresolved count is not excluded). Otherwise the
    status. Counts without a measured compute time are reported separately (candidate_domain)."""
    if status == FEASIBLE:
        return "lower-bound exclusion" if unresolved else "complete enumeration"
    if status == INCUMBENT:
        return "incumbent"
    return status


def candidate_domain(unavailable) -> str:
    return "measured counts only (see unavailable_counts)" if unavailable else "all divisor counts"


def _digest(seq) -> str:
    import hashlib
    return hashlib.sha1(repr(tuple(tuple(c) if isinstance(c, (list, tuple)) else c for c in seq)).encode()).hexdigest()[:12]


class _Accounting:
    """Cached evaluator shared by the initialization and the proposal phase of one search,
    so that every evaluation is counted once and attributed to the phase that caused it."""

    def __init__(self, evaluate):
        self.evaluate = evaluate
        self.cache = {}
        self.phase = "initialization"
        self.counts = {ph: dict(evaluations=0, **{s: 0 for s in STATUSES}) for ph in ("initialization", "proposals")}
        self.cache_hits = 0

    def __call__(self, seq):
        k = tuple(seq)
        if k in self.cache:
            self.cache_hits += 1
            return self.cache[k]
        s, c = self.evaluate(list(seq))
        self.cache[k] = (s, c)
        cnt = self.counts[self.phase]
        cnt["evaluations"] += 1
        cnt[s] = cnt.get(s, 0) + 1
        return self.cache[k]

    def total(self):
        out = {}
        for d in self.counts.values():
            for k, v in d.items():
                out[k] = out.get(k, 0) + v
        return out


def metropolis(evaluate, A, starts, iterations: int, seed: int, beta_scale: float = 0.05, random_start=None):
    """FlexFlow-derived Metropolis search over per-layer configurations.

    evaluate(seq) -> (status, cost). starts: list of (name, seq) initial strategies.
    random_start: optional dict(rng=..., tries=...); a random feasible plan is searched
    for (up to `tries` random plans) and appended as the start "random-feasible".
    All starts are evaluated and the chain starts from the best feasible one (the best
    initial incumbent is preserved). Proposals change one random layer to a random allowed
    configuration and are accepted with probability min(1, exp(-beta (c' - c))),
    beta = 1 / (beta_scale * C_start). A proposal whose evaluation is not established is
    rejected and counted; it is not recorded as infeasible.

    Accounting (F4): one cached evaluator serves the random-start search, the start
    evaluations (phase "initialization") and the proposals (phase "proposals"); every
    distinct plan is evaluated and counted once per search, in the phase that first needed
    it. `counts` is the total of both phases; `evaluated_plan_digests` identifies the
    distinct plans, so that plans shared between seeds can be deduplicated."""
    t0 = time.perf_counter()
    ev = _Accounting(evaluate)
    starts = list(starts)
    tries_used = 0
    if random_start is not None:
        seq, tries_used = random_feasible(ev, A, random_start["rng"], random_start.get("tries", 50),
                                          return_tries=True)
        starts.append(("random-feasible", seq))
    rng = np.random.default_rng(seed)
    start_results, feas_starts = [], []
    for name, seq in starts:
        if seq is None:
            start_results.append(dict(name=name, status="not found", cost=math.inf))
            continue
        s, c = ev(seq)
        start_results.append(dict(name=name, status=s, cost=c))
        if s == FEASIBLE:
            feas_starts.append((c, name, seq))
    t_init = time.perf_counter() - t0

    def acct(extra):
        tot = ev.total()
        return dict(extra, counts=tot, counts_initialization=ev.counts["initialization"],
                    counts_proposals=ev.counts["proposals"], cache_hits=ev.cache_hits,
                    random_start_tries=tries_used, seconds_initialization=t_init,
                    seconds=time.perf_counter() - t0,
                    evaluated_plan_digests=sorted(_digest(k) for k in ev.cache))
    if not feas_starts:
        return acct(dict(seed=seed, best_cost=math.inf, best=None, status=NOT_ESTABLISHED, starts=start_results,
                         iterations=iterations, accepted=0, trace=[]))
    ev.phase = "proposals"
    c0, name0, seq0 = min(feas_starts, key=lambda x: x[0])
    cur, cc = list(seq0), c0
    best, best_seq = c0, list(seq0)
    beta = 1.0 / max(1e-12, beta_scale * c0)
    trace = [(0, ev.total()["evaluations"], time.perf_counter() - t0, best)]
    accepted = 0
    for it in range(1, iterations + 1):
        i = int(rng.integers(len(A)))
        new = list(cur)
        new[i] = A[i][int(rng.integers(len(A[i])))]
        s, c = ev(new)
        if s == FEASIBLE and (c <= cc or rng.random() < math.exp(-beta * (c - cc))):
            cur, cc = new, c
            accepted += 1
            if c < best:
                best, best_seq = c, list(new)
        if it % max(1, iterations // 100) == 0 or it == iterations:
            trace.append((it, ev.total()["evaluations"], time.perf_counter() - t0, best))
    # some proposals not established: chain runner, a declared label cap stopped their evaluation;
    # branch runner, their fastest schedule combination failed the budget without proof (no cap)
    status = FEASIBLE if ev.total().get(NOT_ESTABLISHED, 0) == 0 else "feasible (unresolved proposals)"
    return acct(dict(seed=seed, best_cost=best, best=best_seq, status=status, start=name0, starts=start_results,
                     iterations=iterations, accepted=accepted, trace=trace))


def aggregate_mcmc(runs) -> dict:
    """Totals over the seeds of a multi-seed Metropolis search (F4). `evaluations_sum_per_seed`
    adds each seed's distinct evaluations (a plan evaluated by two seeds counts twice);
    `evaluations_globally_distinct` counts distinct plans over all seeds."""
    tot = lambda key, sub="evaluations": sum((x.get(key) or {}).get(sub, 0) for x in runs)
    digests = set()
    for x in runs:
        digests.update(x.get("evaluated_plan_digests") or [])
    return dict(evaluations_sum_per_seed=tot("counts"), evaluations_initialization_sum=tot("counts_initialization"),
                evaluations_proposals_sum=tot("counts_proposals"), evaluations_globally_distinct=len(digests),
                not_established_evaluations=tot("counts", NOT_ESTABLISHED),
                unavailable_evaluations=tot("counts", UNAVAILABLE),
                seconds_sum=sum(x.get("seconds", 0.0) for x in runs),
                seconds_initialization_sum=sum(x.get("seconds_initialization", 0.0) for x in runs))


def random_feasible(evaluate, A, rng, tries: int = 50, return_tries: bool = False):
    for t in range(1, tries + 1):
        seq = [A[i][int(rng.integers(len(A[i])))] for i in range(len(A))]
        s, _ = evaluate(seq)
        if s == FEASIBLE:
            return (seq, t) if return_tries else seq
    return (None, tries) if return_tries else None


__all__ = ["FEASIBLE", "INFEASIBLE", "NOT_ESTABLISHED", "INCUMBENT", "UNAVAILABLE", "STATUSES", "owt_configs", "summarize_counts", "certification", "candidate_domain", "aggregate_mcmc",
           "metropolis", "random_feasible"]
