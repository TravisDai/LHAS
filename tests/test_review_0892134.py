"""Regressions for the findings of LHAS_0892134_Review.md (and the carried-over findings of
LHAS_a100352_Review.md): F1, F3, F4, F5, T1, T5, T6. Each test reproduces the defect's
failure condition and asserts the corrected behavior."""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lhas.collectives.service import Alt, CATS
from lhas.model.branch import OpChoice
from lhas.model.chain import ChainModel, Settings, DP, MP
from lhas.planner.graph_eval import evaluate_graph_plan_full
from lhas.params import load_approved
from lhas.collectives.service import CollectiveService
from lhas.model.workload import load_workload
from lhas.baselines import chain_baselines as CB
from lhas.baselines import graph_baselines as GB
from lhas.baselines import common as C

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profiles" / "profiled_Tesla_T4_B1024.json"


# ---------------------------------------------------------------- F1
def _alt(name, t, mem, N=2):
    led = {k: np.zeros(N) for k in CATS}
    led["relay"][0] = mem
    return Alt("AG", name, t, 0, led)


def test_f1_nonterminal_mp_keeps_alternatives_and_spec_fixture():
    """Reviewer fixture: fastest alternative reserves 20 bytes, a slower one 5 bytes,
    budget 10. The evaluator does not search slower combinations, so the status must be
    not_established with the 5-byte lower bound, not a false infeasibility proof."""
    N = 2
    fast, slow = _alt("fast", 1, 20), _alt("slow", 2, 5)
    prod = NS(kind="layer", ref="a", ops=[])
    I = NS(join="chain", numel=1, producers=[prod], consumers=["b"], consumer_ops={}, idx=0)
    m = NS(gs=NS(iface_of_output={"a": 0}, interfaces=[I]), N=N, psi=4, B=2, order=["a", "b"], budget=10)
    m.m = lambda n, c: np.zeros(N)
    m.ag_mask = lambda *a: np.ones(N)
    m.rows = lambda *a: 1
    m.charge = lambda n, c, ag=None: ((1, [OpChoice("AG_y", fast, np.ones(N), [fast, slow], ("AG", 2, 2, 4))])
                                      if n == "a" else (0, []))
    m.bp_pair = lambda *a: (0, [])
    m.local_bytes_layer = lambda *a: 4
    m.local_sum_time = lambda *a: 0
    m.local_bytes_input = lambda *a: 4
    r = evaluate_graph_plan_full(m, {"a": (MP, 2), "b": (DP, 2)})
    o = r["ops"][0][1]
    assert o.alts == [fast, slow] and o.spec == ("AG", 2, 2, 4)
    assert r["status"] == "not_established"
    assert r["memory_lower_bound"].tolist() == [5.0, 0.0]


def test_f1_real_graph_inner_mp_ops_are_verifiable():
    from lhas.model.graph import build_structure
    from lhas.model.branch import GraphModel
    from lhas.verify_plan import verify_graph_plan
    tp, cp = load_approved(8)
    gm = GraphModel(build_structure(load_workload("googlenet")), CollectiveService(tp, cp, keep_structures=False),
                    Settings(B=1024, raw_input_grad="omit", norm_policy="sync"))
    cfg = {n: (DP, 8) for n in gm.order}
    inner = "inception5b_branch1_conv"                  # non-terminal layer, output feeds a concatenation
    assert gm.gs.iface_of_output.get(inner) is not None
    cfg[inner] = (MP, 4)
    assert cfg[inner] in gm.configs(inner)
    r = evaluate_graph_plan_full(gm, cfg)
    own = [o for n, o in r["ops"] if n == inner]
    assert {o.name for o in own} >= {"AG_y", "AR_x"}
    assert all(o.alts and o.spec is not None for o in own)
    v = verify_graph_plan(gm, [(n, o) for n, o in r["ops"] if n == inner])
    assert v["summary"]["by_status"].get("skipped", 0) == 0 and v["summary"]["all_network_operations_checked"]


# ---------------------------------------------------------------- F3, T5
@pytest.fixture
def run_gpipe(monkeypatch):
    sys.path.insert(0, str(ROOT / "experiments"))
    import run_gpipe as RG
    captured = {}
    monkeypatch.setattr(RG, "write", lambda a, out: captured.update(out) or "<captured>")
    return RG, captured


@pytest.mark.parametrize("unresolved,expected", [(1, "not_established"), (0, "infeasible")])
def test_f3_gpipe_status_without_incumbent(run_gpipe, monkeypatch, unresolved, expected):
    RG, captured = run_gpipe
    monkeypatch.setattr(RG, "gpipe", lambda *a, **k: (None, dict(candidates=1, evaluated=1, pruned_by_bound=0,
                                                                 allreduce_selection_unresolved=unresolved)))
    RG.main(["--workload", "alexnet", "--N", "4"])
    assert captured["result"]["status"] == expected


def test_t5_gpipe_rejects_profile(run_gpipe, monkeypatch):
    RG, captured = run_gpipe
    monkeypatch.setattr(RG, "gpipe", lambda *a, **k: pytest.fail("GPipe search must not run with --profile"))
    with pytest.raises(SystemExit) as e:
        RG.main(["--workload", "alexnet", "--N", "4", "--profile", str(PROFILE)])
    assert e.value.code == 2 and not captured


# ---------------------------------------------------------------- T6
def _vgg_profiled(N=8):
    tp, cp = load_approved(N)
    return ChainModel(load_workload("vgg16"), CollectiveService(tp, cp), Settings(B=1024, raw_input_grad="omit",
                                                                              optimizer_state_factor=1.0),
                      compute_table=json.loads(PROFILE.read_text()))


def test_t6_chain_missing_profile_is_unavailable_not_infeasible():
    m = _vgg_profiled()
    s = CB.single_node(m)                                  # DP1 of features_0 is unmeasured
    assert s["status"] == C.UNAVAILABLE and "vgg16|features_0|DP|1" in s["missing"]
    r = CB.pure_dp(m, 2)
    assert r.status == C.UNAVAILABLE
    summ, _ = CB.pure_dp_best(m)
    assert set(summ["unavailable_counts"]) >= {1, 2}
    assert all(summ["per_count"][p]["status"] == C.UNAVAILABLE for p in summ["unavailable_counts"])
    assert summ["status"] in (C.FEASIBLE, C.INCUMBENT, C.INFEASIBLE)


def test_t6_graph_missing_profile_is_unavailable():
    from lhas.model.graph import build_structure
    from lhas.model.branch import GraphModel
    t = json.loads(PROFILE.read_text())
    t = copy.deepcopy(t)
    del t["table"]["googlenet|conv1_conv|DP|1"]
    tp, cp = load_approved(8)
    gm = GraphModel(build_structure(load_workload("googlenet")), CollectiveService(tp, cp, keep_structures=False),
                    Settings(B=1024, raw_input_grad="omit", norm_policy="sync"), compute_table=t)
    s = GB.single_node(gm)
    assert s["status"] == C.UNAVAILABLE and s["missing"] == ["googlenet|conv1_conv|DP|1"]
    f = GB.pure_dp(gm, 1)
    assert f["status"] == C.UNAVAILABLE
    summ = GB.pure_dp_best(gm)
    assert summ["unavailable_counts"] == [1] and summ["per_count"][1]["status"] == C.UNAVAILABLE


# ---------------------------------------------------------------- F4
def test_f4_metropolis_accounts_initialization_and_distinct_plans():
    calls = []
    A = [[0, 1, 2]] * 3

    def ev(seq):
        calls.append(tuple(seq))
        return (C.FEASIBLE, 1.0 + sum(seq)) if seq[0] != 2 else (C.INFEASIBLE, math.inf)
    rng = np.random.default_rng(1)
    r = C.metropolis(ev, A, [("s", [1, 1, 1])], 40, seed=0, random_start=dict(rng=rng, tries=50))
    ini, pro = r["counts_initialization"], r["counts_proposals"]
    assert ini["evaluations"] + pro["evaluations"] == r["counts"]["evaluations"] == len(set(calls)) == len(calls)
    assert ini["evaluations"] >= 2                        # random-start tries and the declared start
    assert len(r["evaluated_plan_digests"]) == r["counts"]["evaluations"]
    assert r["seconds"] >= r["seconds_initialization"] >= 0.0
    assert [s["name"] for s in r["starts"]] == ["s", "random-feasible"]


def test_f4_chain_flexflow_reports_initialization():
    tp, cp = load_approved(8)
    m = ChainModel(load_workload("alexnet"), CollectiveService(tp, cp), Settings(B=1024, raw_input_grad="omit"))
    r = CB.flexflow_mcmc(m, iterations=10, seed=0, starts=[("DP(N)", [(DP, 8)] * len(m.L))])
    assert r["counts_initialization"]["evaluations"] >= 2 and "seconds_initialization" in r


# ---------------------------------------------------------------- F5
def test_f5_branch_best_dp_serializes_lower_bounds(monkeypatch):
    from lhas.model.graph import build_structure
    from lhas.model.branch import GraphModel
    tp, cp = load_approved(64)
    gm = GraphModel(build_structure(load_workload("resnet50")), CollectiveService(tp, cp, keep_structures=False),
                    Settings(B=1024, raw_input_grad="omit", norm_policy="sync"))
    real = GB.pure_dp

    def fake(model, p):
        f = real(model, p)
        if p == 2:                                    # force one unresolved count with a known bound
            f = dict(f, status=C.NOT_ESTABLISHED, cost=math.inf, cost_if_feasible=123.0)
        if p == 4:                                    # and one whose bound does not exclude it
            f = dict(f, status=C.NOT_ESTABLISHED, cost=math.inf, cost_if_feasible=1e-9)
        return f
    monkeypatch.setattr(GB, "pure_dp", fake)
    summ = GB.pure_dp_best(gm)
    assert summ["per_count"][2]["lower_bound"] == 123.0 and summ["per_count"][4]["lower_bound"] == 1e-9
    assert summ["status"] == C.INCUMBENT and summ["certification"] == "incumbent"
    assert summ["unresolved_counts"] == [2, 4]


# ---------------------------------------------------------------- T1
def test_t1_grouped_split_has_no_shared_kernel_signatures():
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyse_profile as AP
    d = AP.load(AP.ZIP)
    rows = [r for r in d["rows"] if r["status"] == "measured"]
    groups = AP.group_by_signature(rows)
    cal, val = AP.split_groups(sorted(groups), seed=0)
    assert not (set(cal) & set(val)) and len(cal) + len(val) == len(groups)
    sig_of_id = {r["shape_id"]: AP.kernel_signature(r) for r in rows}
    cal_ids = {i for s in cal for i in (r["shape_id"] for r in groups[s])}
    val_ids = {i for s in val for i in (r["shape_id"] for r in groups[s])}
    assert not ({sig_of_id[i] for i in cal_ids} & {sig_of_id[i] for i in val_ids})
    assert "n_out_full" not in AP.SIGNATURE_FIELDS
    assert len(groups) == 1542                           # 1759 measured records, 1542 timed kernel shapes
