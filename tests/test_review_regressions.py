"""Regression tests for the defects reported in LHAS_Decision_Answers_and_Code_Review.md
(reproduced at commit 757bd4c by scripts/diagnostics/review_repro.py) and for the
final D1-D17 specification."""
import math

import numpy as np
import pytest

from lhas.params import load_approved
from lhas.model.workload import load_workload
from lhas.model.graph import build_structure
from lhas.model.chain import ChainModel, Settings, DP, MP
from lhas.model.branch import GraphModel
from lhas.model import layercost as LC
from lhas.collectives.service import CollectiveService
from lhas.planner.chain_dp import plan_chain
from lhas.planner.graph_eval import evaluate_graph_plan_full
from lhas.baselines import chain_baselines as CB
from lhas.baselines import graph_baselines as GB
from lhas.baselines import common as C

NOM = dict(B=1024, raw_input_grad="omit", optimizer_state_factor=1.0)
GiB = 2 ** 30


def chain(wl="alexnet", N=16, reserve=GiB, **kw):
    tp, cp = load_approved(N)
    from dataclasses import replace
    cp = replace(cp, runtime_reserve_bytes=reserve)
    st = Settings(**{**NOM, **kw})
    return ChainModel(load_workload(wl), CollectiveService(tp, cp), st)


def graph(wl, N=16, reserve=GiB, norm="sync", **kw):
    tp, cp = load_approved(N)
    from dataclasses import replace
    cp = replace(cp, runtime_reserve_bytes=reserve)
    st = Settings(**{**NOM, "norm_policy": norm, **kw})
    return GraphModel(build_structure(load_workload(wl)), CollectiveService(tp, cp, keep_structures=False), st)


# ---------------------------------------------------------------- greedy ablation
def test_greedy_local_charge_uses_its_own_layer_payloads():
    m = chain("alexnet", N=64)
    i, c = 6, (MP, 16)
    assert c in m.configs(i) and c not in m.configs(i + 1)   # 16 does not divide 1000 outputs
    calls = []
    ag0, ar0 = m.svc.allgather, m.svc.allreduce
    m.svc.allgather = lambda p, q, V: calls.append(("AG", p, q, V)) or ag0(p, q, V)
    m.svc.allreduce = lambda p, V: calls.append(("AR", p, V)) or ar0(p, V)
    m.local_charge(i, c)
    L = m.L[i]
    assert ("AG", 16, 16, m.psi * L.u_numel * 1024 // 16) in calls
    last = m.L[-1]
    assert all(x[-1] != m.psi * last.y_numel * 1024 // 16 for x in calls if x[0] == "AG")
    seq, r = CB.greedy_layerwise(m)
    assert len(seq) == len(m.L) and all(s in m.configs(k) for k, s in enumerate(seq))


# ---------------------------------------------------------------- aliasing rules
def _residual_node1(model, cfg):
    r = evaluate_graph_plan_full(model, cfg)
    base = sum(model.m(n, cfg[n]) for n in model.order)
    ops = sum((o.mem() for _, o in r["ops"]), np.zeros(model.N))
    return r, (r["memory"] - base - ops)


def test_resnet_final_interface_charges_prepooling_operands():
    gm = graph("resnet50")
    gs = gm.gs
    I = next(I for I in gs.interfaces if "fc" in I.consumers)
    assert I.join == "add" and I.consumer_ops["fc"][-2:] == ["adaptive_avgpool", "flatten"]
    assert gm.L["fc"].x_numel < I.numel                    # 2048 vs 2048x7x7
    cfg = {n: (DP, 16) for n in gm.order}
    prod = next(p for p in I.producers if p.kind == "layer").ref
    cfg[prod] = (MP, 4)
    cfg["fc"] = (MP, 4)
    assert all(cfg[n] in gm.configs(n) for n in gm.order)
    _, res = _residual_node1(gm, cfg)
    V = gm.psi * I.numel * 1024                            # complete pre-pooling tensor at an MP consumer
    assert gm.local_bytes_input(I, cfg["fc"]) == V
    assert not gm.shape_preserving_add(I, "fc")
    assert res[0] >= 3 * V - 1                              # J1: both operands and the pre-pooling gradient
    # the assembled All-gather at node 1 is covered by that explicitly charged buffer only
    assert gm.consumer_full_T_buffer(I, "fc") == "J1"


def test_googlenet_consumer_side_pooling_is_charged_C1():
    gm = graph("googlenet")
    I = next(I for I in gm.gs.interfaces if any(I.consumer_ops.get(c) for c in I.consumers) and I.join != "input")
    c4 = next(c for c in I.consumers if I.consumer_ops[c])
    assert I.consumer_ops[c4] == ["maxpool"]
    assert gm.consumer_full_T_buffer(I, c4) == "C1"
    cfg = {n: (DP, 16) for n in gm.order}
    cfg[c4] = (MP, 4)
    assert cfg[c4] in gm.configs(c4)
    _, res = _residual_node1(gm, cfg)
    V = gm.local_bytes_input(I, (MP, 4))
    assert res[0] >= 2 * V - 1
    # a consumer without consumer-side ops and equal element count aliases into its own input
    c1 = next(c for c in I.consumers if not I.consumer_ops[c])
    assert gm.consumer_full_T_buffer(I, c1) == ("x" if gm.L[c1].x_numel == I.numel else None)


def test_producer_alias_only_for_inplace_relu():
    assert LC.aliasable_ops(["relu"]) and LC.aliasable_ops([])
    for ops in (["relu", "dropout"], ["maxpool"], ["batchnorm"], ["flatten"]):
        assert not LC.aliasable_ops(ops)
    wl = load_workload("alexnet")
    fc6 = next(L for L in wl.layers if L.name == "classifier_1")
    assert "dropout" in fc6.phi_ops and not fc6.phi_identity


# ---------------------------------------------------------------- BatchNorm state (D4, D8)
def test_batchnorm_state_bytes_and_int64_counter():
    L = next(L for L in load_workload("resnet50").layers if L.norm)
    C_ = L.norm["num_features"]
    assert L.norm["buffers"]["num_batches_tracked"] == 1
    assert LC.norm_state_bytes(L, DP, 8, 4, "sync") == 4 * 4 * C_ + 8
    assert LC.norm_state_bytes(L, MP, 4, 4, "sync") == pytest.approx(4 * 4 * C_ / 4 + 8)
    assert LC.norm_state_bytes(L, DP, 8, 4, "none") == 0
    a = LC.layer_reservation(L, (MP, 4), 1024, 4, "sync", 1.0, 0, 8)
    b = LC.layer_reservation(L, (MP, 4), 1024, 4, "none", 1.0, 0, 8)
    params = L.norm["params"]
    assert a[0] - b[0] == pytest.approx(4 * (2 + 1.0) * params / 4 + 4 * 4 * C_ / 4 + 8)
    assert np.all(a[4:] == 0)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_sync_batchnorm_two_reductions_equal_full_batch(p):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    Bt, Ct, H = 8, 6, 5
    x = torch.randn(Bt, Ct, H, H, dtype=torch.float64, requires_grad=True)
    g = torch.randn(Ct, dtype=torch.float64, requires_grad=True)
    b = torch.randn(Ct, dtype=torch.float64, requires_grad=True)
    dy = torch.randn(Bt, Ct, H, H, dtype=torch.float64)
    bn = torch.nn.functional.batch_norm(x, None, None, g, b, training=True, eps=1e-5)
    bn.backward(dy)
    ref = (bn.detach(), x.grad.clone(), g.grad.clone(), b.grad.clone())
    # DP(p): forward All-reduce of (sum x, sum x^2) = 2C; backward All-reduce of
    # (sum dy, sum dy*xhat) = 2C; everything else is shard-local
    xs, dys = x.detach().chunk(p), dy.chunk(p)
    n = Bt * H * H
    s1 = sum(t.sum((0, 2, 3)) for t in xs)
    s2 = sum((t * t).sum((0, 2, 3)) for t in xs)
    mean = s1 / n
    var = s2 / n - mean ** 2
    inv = 1.0 / torch.sqrt(var + 1e-5)
    sh = lambda v: v.view(1, -1, 1, 1)  # noqa: E731
    xh = [(t - sh(mean)) * sh(inv) for t in xs]
    y = torch.cat([h * sh(g.detach()) + sh(b.detach()) for h in xh])
    sdy = sum(t.sum((0, 2, 3)) for t in dys)
    sdyx = sum((t * h).sum((0, 2, 3)) for t, h in zip(dys, xh))
    dx = torch.cat([sh(g.detach() * inv) * (t - sh(sdy / n) - h * sh(sdyx / n)) for t, h in zip(dys, xh)])
    assert torch.allclose(y, ref[0], atol=1e-10)
    assert torch.allclose(dx, ref[1], atol=1e-10)
    assert torch.allclose(sdyx, ref[2], atol=1e-10) and torch.allclose(sdy, ref[3], atol=1e-10)
    # MP (channel shards over the complete batch): channel-local, no reduction
    for cs in torch.arange(Ct).chunk(p):
        xc = x.detach()[:, cs].clone().requires_grad_(True)
        yc = torch.nn.functional.batch_norm(xc, None, None, g.detach()[cs], b.detach()[cs], training=True, eps=1e-5)
        yc.backward(dy[:, cs])
        assert torch.allclose(yc, ref[0][:, cs], atol=1e-10) and torch.allclose(xc.grad, ref[1][:, cs], atol=1e-10)


# ---------------------------------------------------------------- reserve (D7)
def test_runtime_reserve_charged_once_per_node():
    assert chain("alexnet", N=8).budget == 12 * GiB - GiB
    assert graph("alexnet", N=8, norm="none").budget == 12 * GiB - GiB
    assert chain("alexnet", N=8, reserve=0).budget == 12 * GiB


# ---------------------------------------------------------------- statuses (D10, D11)
def test_label_cap_is_not_established_not_infeasible():
    m = chain("alexnet", N=64)
    seq = [(DP, 64)] * len(m.L)
    r = CB.evaluate_fixed(m, seq, max_labels=1)
    assert r.status == C.NOT_ESTABLISHED and not r.feasible and r.stats.get("not_completed")
    full = CB.evaluate_fixed(m, seq)
    assert r.stats["lower_bound"] <= full.cost + 1e-12
    assert CB.evaluate_fixed(m, seq).status == C.FEASIBLE
    # a proven infeasible sequence (layer reservations alone exceed the budget) stays infeasible
    v = chain("vgg16", N=16)
    assert CB.evaluate_fixed(v, [(DP, 16)] * len(v.L)).status == C.INFEASIBLE


def test_summarize_counts_incumbent_semantics():
    F, I, U = C.FEASIBLE, C.INFEASIBLE, C.NOT_ESTABLISHED
    assert C.summarize_counts({1: (F, 3.0), 2: (F, 2.0), 4: (I, math.inf)})[:3] == (F, 2, 2.0)
    assert C.summarize_counts({1: (F, 3.0), 2: (U, math.inf)})[:3] == (C.INCUMBENT, 1, 3.0)
    assert C.summarize_counts({1: (I, math.inf), 2: (U, math.inf)})[0] == U
    assert C.summarize_counts({1: (I, math.inf)})[0] == I
    # an unresolved count whose lower bound is no better than the best completed result is dominated
    assert C.summarize_counts({1: (F, 3.0), 2: (U, math.inf)}, {2: 3.5})[0] == F
    assert C.summarize_counts({1: (F, 3.0), 2: (U, math.inf)}, {2: 2.5})[0] == C.INCUMBENT


def test_pure_dp_best_reports_status_and_all_counts():
    m = chain("alexnet", N=16)
    summ, best = CB.pure_dp_best(m)
    assert summ["status"] in (C.FEASIBLE, C.INCUMBENT)
    assert set(summ["per_count"]) == {p for p in range(1, 17) if 1024 % p == 0}
    assert best.cost == summ["cost"] == min(v["cost"] for v in summ["per_count"].values() if v["status"] == C.FEASIBLE)


def test_graph_fixed_plan_statuses():
    gm = graph("googlenet", N=16)
    f = GB.fixed(gm, {n: (DP, 16) for n in gm.order})
    assert f["status"] == C.FEASIBLE and math.isfinite(f["cost"])
    single = GB.single_node(gm)
    assert single["status"] in (C.INFEASIBLE, C.NOT_ESTABLISHED) and single["cost"] == math.inf
    assert single["time_only"] > 0


# ---------------------------------------------------------------- OWT (D12)
@pytest.mark.parametrize("N", [16, 64])
def test_owt_single_definition_in_both_runners(N):
    m = chain("alexnet", N=N)
    gm = graph("alexnet", N=N, norm="none")
    seq_c, _ = CB.owt(m)
    seq_g = C.owt_configs([gm.L[n] for n in gm.order], N, 1024)
    assert seq_c == seq_g
    for L, c in zip(m.L, seq_c):
        if L.kind == "conv":
            assert c == (DP, N)
        else:
            assert c[0] == MP and L.n_out % c[1] == 0 and all(L.n_out % q for q in range(c[1] + 1, N + 1))
    assert C.owt_configs(m.L, 48, 1024) is None


# ---------------------------------------------------------------- Metropolis search (D14)
def test_metropolis_multistart_and_unresolved_counting():
    A = [[0, 1, 2]] * 4

    def ev(seq):
        if seq[0] == 2:
            return C.NOT_ESTABLISHED, math.inf
        if seq[1] == 2:
            return C.INFEASIBLE, math.inf
        return C.FEASIBLE, 1.0 + sum(seq)
    r = C.metropolis(ev, A, [("worse", [1, 1, 1, 1]), ("better", [0, 0, 0, 1]), ("none", None)], 300, seed=0)
    assert r["start"] == "better" and r["best_cost"] == 1.0 and r["best"] == [0, 0, 0, 0]
    assert r["counts"]["not_established"] > 0 and r["status"] == "feasible (unresolved proposals)"
    tr = [t[3] for t in r["trace"]]
    assert all(a >= b for a, b in zip(tr, tr[1:]))
    assert [x["name"] for x in r["starts"]] == ["worse", "better", "none"]
    assert r["starts"][2]["status"] == "not found"
    assert r["counts"]["evaluations"] == sum(r["counts"][k] for k in ("feasible", "infeasible", "not_established"))


# ---------------------------------------------------------------- GPipe (D13)
def test_gpipe_simulation_matches_closed_form_for_equal_stages():
    from lhas.baselines.gpipe import simulate_pipeline, MICRO
    assert MICRO[:2] == (1, 2)
    for S, M in [(1, 1), (3, 4), (4, 8)]:
        tf, tb = [1.0] * S, [2.0] * S
        f, e = simulate_pipeline(tf, tb, [0.0] * (S - 1), [0.0] * (S - 1), M)
        assert f == pytest.approx((M + S - 1) * 1.0) and e == pytest.approx((M + S - 1) * 3.0)
    # transfers delay but never shorten the makespan
    f2, e2 = simulate_pipeline([1.0] * 3, [2.0] * 3, [0.5, 0.5], [0.5, 0.5], 4)
    assert e2 > 18.0


def test_gpipe_contract_small():
    from lhas.baselines.gpipe import gpipe
    m = chain("alexnet", N=8)
    best, stats = gpipe(m)
    assert best is not None and best.status in ("feasible", "incumbent")
    assert best.P == best.R * best.S <= 8 and 1024 % (best.R * best.M) == 0
    assert best.memory_max <= m.budget + 1e-6
    assert best.cost == pytest.approx(best.breakdown["forward"] + best.breakdown["backward"]
                                      + best.breakdown["setup"] + best.breakdown["allreduce"])
    d, _ = gpipe(m, stage_counts=[1])
    assert d.degenerate_dp and d.S == 1
    stats_keys = {"candidates", "evaluated", "pruned_by_bound", "allreduce_selection_unresolved"}
    assert stats_keys <= set(stats)


def test_gpipe_allreduce_ring_translation():
    m = chain("alexnet", N=16)
    base = tuple(r * 4 + 1 for r in range(4))
    shifted = tuple(x + 2 for x in base)
    a = m.svc.allreduce_list(base, 3_000_000)
    b = m.svc.allreduce_list(shifted, 3_000_000)
    assert [x.latency for x in a] == pytest.approx([x.latency for x in b])
    for x, y in zip(a, b):
        assert np.allclose(np.roll(x.mem(), 2), y.mem())


# ---------------------------------------------------------------- verification coverage
def test_verification_records_every_selected_operation(monkeypatch):
    from lhas import verify_plan as VP
    m = chain("alexnet", N=8)
    r = plan_chain(m)
    out = VP.verify_chain_plan(m, r)
    s = out["summary"]
    assert s["expected_operations"] == len(out["operations"]) == sum(s["by_status"].values())
    assert s["all_network_operations_checked"]
    monkeypatch.setattr(VP, "MAX_TRANSMISSIONS", 1)
    monkeypatch.setattr(VP, "MAX_VERIFY_ROUTES", 3)
    out2 = VP.verify_chain_plan(m, r)
    st = {x["status"] for x in out2["operations"]}
    assert "skipped" in st or "checked-reduced" in st
    for x in out2["operations"]:
        if x["status"] in ("skipped", "checked-reduced"):
            assert x["reason"]
    assert not out2["summary"]["all_network_operations_checked"] or "skipped" not in st
