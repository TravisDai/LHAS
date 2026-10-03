"""Typed-interface exact search versus brute force on a small synthetic graph
with a three-way fork (one branch with consumer-side pooling), a concatenation,
and a residual addition with an identity operand. Brute force enumerates every
configuration of every layer and evaluates each plan with the independent
plan evaluator (same min-charge alternative semantics)."""
import itertools
import json

import numpy as np
import pytest

from lhas.params import load_approved
from lhas.collectives.service import CollectiveService
from lhas.model.workload import load_workload
from lhas.model.graph import build_structure
from lhas.model.chain import Settings
from lhas.model.branch import GraphModel
from lhas.planner.graph_dp import plan_graph
from lhas.planner.graph_eval import evaluate_graph_plan


def conv(id_, inp, cin, cout, hw):
    return dict(id=id_, op="conv", inputs=[inp], shape=[cout, hw, hw], in_channels=cin, out_channels=cout,
                kernel=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1,
                param_numel=dict(weight=cin * cout, bias=cout), buffer_numel={})


def node(id_, op, inputs, shape, **kw):
    return dict(id=id_, op=op, inputs=inputs, shape=shape, param_numel={}, buffer_numel={}, **kw)


def manifest(hw=3):
    n = [node("x", "input", [], [3, hw, hw]),
         conv("c1", "x", 3, 4, hw), node("r1", "relu", ["c1"], [4, hw, hw]),
         conv("c2", "r1", 4, 4, hw), node("r2", "relu", ["c2"], [4, hw, hw]),
         conv("c3", "r1", 4, 2, hw), node("r3", "relu", ["c3"], [2, hw, hw]),
         node("mp", "maxpool", ["r1"], [4, hw, hw]),
         conv("c5", "mp", 4, 2, hw), node("r5", "relu", ["c5"], [2, hw, hw]),
         node("cat", "cat", ["r2", "r3", "r5"], [8, hw, hw], dim=1),
         conv("c6", "cat", 8, 4, hw), node("r6", "relu", ["c6"], [4, hw, hw]),
         conv("c7", "r6", 4, 4, hw),
         node("add", "add", ["c7", "r6"], [4, hw, hw]), node("r7", "relu", ["add"], [4, hw, hw]),
         node("fl", "flatten", ["r7"], [4 * hw * hw], start_dim=1),
         dict(id="fc", op="linear", inputs=["fl"], shape=[4], in_features=4 * hw * hw, out_features=4,
              param_numel=dict(weight=4 * 4 * hw * hw, bias=4), buffer_numel={}),
         node("output", "output", ["fc"], [4])]
    return dict(workload="toybranch", nodes=n)


@pytest.mark.parametrize("alpha", [0.0, 0.6])
def test_graph_search_equals_brute_force(tmp_path, alpha):
    (tmp_path / "toybranch.json").write_text(json.dumps(manifest()))
    wl = load_workload("toybranch", tmp_path)
    gs = build_structure(wl)
    assert {I.join for I in gs.interfaces} >= {"cat", "add"}
    tp, cp = load_approved(4)
    st = Settings(B=2, alpha=alpha, raw_input_grad="retain")
    model = GraphModel(gs, CollectiveService(tp, cp), st)
    r = plan_graph(model)
    names = model.order
    A = [model.configs(n) for n in names]
    best = min(evaluate_graph_plan(model, dict(zip(names, combo)))[0] for combo in itertools.product(*A))
    assert r.cost == pytest.approx(best, rel=1e-12, abs=1e-15)
