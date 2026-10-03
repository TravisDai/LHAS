"""Typed-interface engine: consistency with the chain planner on chain networks,
search objective equal to independent re-evaluation, structure checks."""
import pytest

from lhas.params import load_approved
from lhas.model.workload import load_workload
from lhas.model.graph import build_structure
from lhas.model.chain import ChainModel, Settings
from lhas.model.branch import GraphModel
from lhas.collectives.service import CollectiveService
from lhas.planner.chain_dp import plan_chain
from lhas.planner.graph_dp import plan_graph


@pytest.mark.parametrize("wl,N", [("alexnet", 8), ("alexnet", 32), ("vgg16", 32)])
@pytest.mark.parametrize("raw", ["retain", "omit"])
def test_graph_engine_equals_chain_planner(wl, N, raw):
    tp, cp = load_approved(N)
    W = load_workload(wl)
    st = Settings(B=1024, raw_input_grad=raw)
    rc = plan_chain(ChainModel(W, CollectiveService(tp, cp), st))
    rg = plan_graph(GraphModel(build_structure(W), CollectiveService(tp, cp), st))
    assert rc.feasible and rg.memory_feasible
    assert rg.cost == pytest.approx(rc.cost, rel=1e-12)


@pytest.mark.parametrize("wl", ["googlenet", "resnet50"])
def test_branch_structure_and_consistency(wl):
    tp, cp = load_approved(16)
    W = load_workload(wl)
    gs = build_structure(W)
    joins = {I.join for I in gs.interfaces}
    assert ("cat" in joins) if wl == "googlenet" else ("add" in joins)
    if wl == "googlenet":
        # concatenated slices tile the channel range in input order
        for I in gs.interfaces:
            if I.join == "cat":
                offs = [(p.ch0, p.ch1) for p in I.producers]
                assert offs[0][0] == 0 and offs[-1][1] == I.channels
                assert all(a[1] == b[0] for a, b in zip(offs, offs[1:]))
    else:
        ids = [p for I in gs.interfaces for p in I.producers if p.kind == "iface"]
        assert len(ids) == 12          # identity shortcuts in ResNet-50 (16 blocks, 4 projections)
    st = Settings(B=1024, raw_input_grad="retain", norm_policy="sync")
    r = plan_graph(GraphModel(gs, CollectiveService(tp, cp), st))
    assert r.cost == pytest.approx(r.stats["search_cost"], rel=1e-12)
