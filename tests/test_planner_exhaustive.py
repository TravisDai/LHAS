"""Chain DP versus exhaustive enumeration on genuinely small instances.

Exhaustive enumeration uses every configuration sequence and *every* schedule
combination (no Pareto filtering), including the terminal operation's
alternatives and reservations. The DP uses Pareto-filtered alternatives, safe
labels and branch-and-bound. Budgets are swept from infeasible to slack so that
the memory constraint binds in some cases and slower schedules are required.
"""
from dataclasses import replace

import numpy as np
import pytest

from lhas.params import load_approved
from lhas.collectives.service import CollectiveService
from lhas.model.chain import ChainModel, Settings
from lhas.model.workload import synthetic_chain
from lhas.planner.chain_dp import plan_chain, exhaustive_chain

SPECS = {
    "fc3": [dict(kind="fc", n_in=24, n_out=12, x_shape=(24,), y_shape=(12,)),
            dict(kind="fc", n_in=12, n_out=36, x_shape=(12,), y_shape=(36,)),
            dict(kind="fc", n_in=36, n_out=6, x_shape=(36,), y_shape=(6,))],
    "conv_fc": [dict(kind="conv", n_in=3, n_out=4, kernel=(3, 3), x_shape=(3, 6, 6), y_shape=(4, 4, 4),
                     u_shape=(4, 2, 2), phi_ops=["relu", "maxpool"]),
                dict(kind="conv", n_in=4, n_out=6, kernel=(1, 1), x_shape=(4, 2, 2), y_shape=(6, 2, 2),
                     u_shape=(24,), phi_ops=["relu", "flatten"]),
                dict(kind="fc", n_in=24, n_out=8, x_shape=(24,), y_shape=(8,))],
    "fc4": [dict(kind="fc", n_in=8, n_out=16, x_shape=(8,), y_shape=(16,)),
            dict(kind="fc", n_in=16, n_out=4, x_shape=(16,), y_shape=(4,)),
            dict(kind="fc", n_in=4, n_out=12, x_shape=(4,), y_shape=(12,)),
            dict(kind="fc", n_in=12, n_out=4, x_shape=(12,), y_shape=(4,))],
}


def build(name, N, B, budget, lam=2, w=2, reach=2, alpha=0.0, raw="retain", kappa=4096, opt=1.0):
    tp, cp = load_approved(N)
    tp = replace(tp, lam=lam, w_tx=w, w_rx=w, reach=reach, d_peer=3, kappa_bytes=kappa)
    cp = replace(cp, budget_bytes=int(budget), runtime_reserve_bytes=0)
    wl = synthetic_chain(name, SPECS[name])
    return ChainModel(wl, CollectiveService(tp, cp),
                      Settings(B=B, alpha=alpha, raw_input_grad=raw, optimizer_state_factor=opt))


def budgets(model):
    """Budgets from below the leanest plan to above the largest plan."""
    lo = min(model.m(0, c).max() for c in model.configs(0, memory_filter=False))
    hi = sum(max(model.m(i, c).max() for c in model.configs(i, memory_filter=False)) for i in range(len(model.L)))
    return np.unique(np.linspace(lo * 0.9, hi * 3.0, 9).astype(np.int64))


@pytest.mark.parametrize("name,N,B", [("fc3", 4, 8), ("conv_fc", 4, 4), ("fc4", 4, 4), ("fc3", 6, 6)])
@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_dp_matches_exhaustive(name, N, B, alpha):
    base = build(name, N, B, 1 << 40, alpha=alpha)
    bound_hit = 0
    for bud in budgets(base):
        m = build(name, N, B, bud, alpha=alpha)
        ex_cost, ex_seq = exhaustive_chain(m, pareto_alts=False)
        for force in (False, True):
            m2 = build(name, N, B, bud, alpha=alpha)
            r = plan_chain(m2, force_labels=force)
            if ex_seq is None:
                assert not r.feasible
            else:
                assert r.feasible
                assert r.cost == pytest.approx(ex_cost, rel=1e-12, abs=1e-15)
                assert np.all(r.memory <= m2.budget + 1e-6)
                if "labels" in r.stats["mode"]:
                    bound_hit += 1
    # the sweep must include memory-binding cases
    assert bound_hit > 0


def test_slower_schedule_required_when_fastest_does_not_fit():
    """Construct a budget between the fastest plan's reservation and a slower
    feasible plan's reservation, and check the DP returns the slower plan."""
    name, N, B = "fc3", 4, 8
    free = build(name, N, B, 1 << 40, lam=1, w=1, reach=1)
    r0 = plan_chain(free)
    need0 = r0.memory.max()
    found = False
    for frac in np.linspace(0.999, 0.5, 40):
        m = build(name, N, B, int(need0 * frac), lam=1, w=1, reach=1)
        ex_cost, ex_seq = exhaustive_chain(m, pareto_alts=False)
        if ex_seq is None:
            break
        r = plan_chain(m)
        assert r.feasible and r.cost == pytest.approx(ex_cost, rel=1e-12)
        if r.cost > r0.cost * (1 + 1e-9):
            found = True
            break
    assert found, "no budget made the fastest plan infeasible while a slower plan fit"


def test_raw_input_gradient_flag_changes_first_layer_only():
    m_keep = build("fc3", 4, 8, 1 << 40, raw="retain")
    m_omit = build("fc3", 4, 8, 1 << 40, raw="omit")
    assert m_omit.flops(0) == pytest.approx(m_keep.flops(0) * 2 / 3)
    assert m_omit.flops(1) == m_keep.flops(1)
    ops_keep = [n for n, *_ in m_keep._intra_ops(0, (1, 2), 2)]
    ops_omit = [n for n, *_ in m_omit._intra_ops(0, (1, 2), 2)]
    assert "AR_x" in ops_keep and "AR_x" not in ops_omit
    assert "AR_x" in [n for n, *_ in m_omit._intra_ops(1, (1, 2), 2)]
