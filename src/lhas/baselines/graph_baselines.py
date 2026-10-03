"""Typed-interface (branch and chain) baselines with the same evaluator as LHAS's
typed-interface search: every operation at its minimum-charge alternative, then the
corrected reservation ledger. Statuses: feasible; infeasible when even the
reservation lower bound over all retained alternatives exceeds the budget;
not_established otherwise (a slower combination might fit, D10)."""
from __future__ import annotations

import math

import numpy as np

from ..model.branch import GraphModel
from ..model.layercost import DP, MP
from ..planner.graph_eval import evaluate_graph_plan_full
from . import common as C


def fixed(model: GraphModel, cfg: dict) -> dict:
    configs = [_s(cfg[n]) for n in model.order]
    missing = model.missing_profile(cfg)
    if missing:                                           # T6: not evaluated, not a memory statement
        return dict(status=C.UNAVAILABLE, cost=math.inf, reason="missing compute profile", missing=missing,
                    configs=configs)
    if not all(model.memory_allowed(n, cfg[n]) for n in model.order):
        return dict(status=C.INFEASIBLE, cost=math.inf, reason="a layer reservation alone exceeds the budget "
                                                               "or a count is not a divisor", configs=configs)
    r = evaluate_graph_plan_full(model, cfg)
    status = r["status"]
    return dict(status=status, cost=r["cost"] if status == C.FEASIBLE else math.inf, cost_if_feasible=r["cost"],
                memory_max=float(r["memory"].max()), memory_lower_bound_max=float(r["memory_lower_bound"].max()),
                configs=configs)


def _s(c):
    return ("DP" if c[0] == DP else "MP") + str(c[1])


def single_node(model):
    cfg = {n: (DP, 1) for n in model.order}
    missing = model.missing_profile(cfg)
    if missing:
        return dict(status=C.UNAVAILABLE, time_only=None, cost=math.inf, reason="missing compute profile",
                    missing=missing)
    r = evaluate_graph_plan_full(model, cfg)
    return dict(status=r["status"], time_only=r["cost"], cost=r["cost"] if r["status"] == C.FEASIBLE else math.inf,
                memory_node1=float(r["memory"][0]))


def pure_dp(model, p):
    return fixed(model, {n: (DP, p) for n in model.order})


def pure_dp_best(model):
    res, plans, lbs, missing = {}, {}, {}, {}
    for p in [p for p in range(1, model.N + 1) if model.B % p == 0]:
        f = pure_dp(model, p)
        res[p] = (f["status"], f["cost"])
        plans[p] = f
        if f["status"] == C.NOT_ESTABLISHED:
            lbs[p] = f["cost_if_feasible"]        # min-charge objective: a lower bound for this count
        if f["status"] == C.UNAVAILABLE:
            missing[p] = f["missing"]
    status, bp, bc, unresolved = C.summarize_counts(res, lbs)
    per = {}
    for k, v in res.items():
        e = dict(status=v[0], cost=v[1])
        if k in lbs:                              # F5: serialize the supporting lower bound
            e.update(lower_bound=lbs[k], lower_bound_kind="minimum-charge additive objective")
        if k in missing:
            e.update(reason="missing compute profile", missing=missing[k])
        per[int(k)] = e
    return dict(status=status, p=bp, cost=bc, unresolved_counts=unresolved, unavailable_counts=sorted(missing),
                certification=C.certification(status, unresolved, missing),
                candidate_domain=C.candidate_domain(missing), per_count=per,
                configs=plans[bp]["configs"] if bp is not None else None)


def owt(model):
    seq = C.owt_configs([model.L[n] for n in model.order], model.N, model.B)
    if seq is None:
        return None
    return fixed(model, dict(zip(model.order, seq)))


def graph_evaluator(model):
    names = model.order

    def ev(seq):
        f = fixed(model, dict(zip(names, seq)))
        return f["status"], f["cost"]
    return ev


def flexflow_mcmc(model, iterations=1000, seed=0, starts=None):
    A = [model.configs(n) for n in model.order]
    ev = graph_evaluator(model)
    rng = np.random.default_rng(10_000 + seed)
    # the random feasible start is searched inside the search's accounting (F4)
    return C.metropolis(ev, A, list(starts or []), iterations, seed, random_start=dict(rng=rng, tries=50))


__all__ = ["fixed", "single_node", "pure_dp", "pure_dp_best", "owt", "flexflow_mcmc", "graph_evaluator"]
