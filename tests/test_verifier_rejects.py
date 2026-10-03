"""Deliberately invalid schedules must be rejected by the independent verifier.

Each test mutates a valid exported record (or builds one by hand) and asserts
the verifier rejects it with the violated rule code. Physical and semantic
checks are exercised with first-fit conformance disabled (check_packing=False)
so that they are shown to work on arbitrary, externally supplied schedules.
"""
import numpy as np
import pytest

from lhas.collectives.families import allgather_ops, wrht_op, osm
from lhas.transport.export import spec_allgather, spec_allreduce, spec_alltoall, params_dict
from lhas_verify import verify, VerificationError
from helpers import stressed, rec_and_params, deep, BETA


def rejects(code, rec, spec, prm, **kw):
    kw.setdefault("check_packing", False)
    with pytest.raises(VerificationError, match=rf"\[{code}\]"):
        verify(rec, spec, prm, **kw)


def strip_claims(rec):
    rec = deep(rec)
    rec.pop("claimed", None)
    for ph in rec["phases"]:
        for rnd in ph["rounds"]:
            for k in ("setup", "steps", "body"):
                rnd.pop(k, None)
    return rec


def all_routes(rec):
    for pi, ph in enumerate(rec["phases"]):
        for qi, rnd in enumerate(ph["rounds"]):
            for ri, r in enumerate(rnd["routes"]):
                yield pi, qi, ri, r


@pytest.fixture(scope="module")
def ag():
    tp = stressed(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000)
    op = [o for o in allgather_ops(12, 4, tp) if o.candidate.startswith("C-A")][0]
    V = 2500                                   # three chunks with a 500-byte tail
    rec, prm = rec_and_params(op, V, tp)
    spec = spec_allgather(range(1, 13), range(1, 5), V)
    verify(rec, spec, prm)                     # the unmodified record is valid
    return rec, prm, spec


@pytest.fixture(scope="module")
def relayed(ag):
    rec, prm, spec = ag
    for pi, qi, ri, r in all_routes(rec):
        if len(r["hops"]) >= 2:
            return pi, qi, ri
    raise AssertionError("no relayed route in fixture")


# ---------------------------------------------------------------- R1 / R2
def test_label_out_of_range(ag):
    rec, prm, spec = ag
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["routes"][0]["hops"][0][4] = prm["lam"] + 1
    rejects("R1", bad, spec, prm)


def test_reach_exceeded(ag):
    rec, prm, spec = ag
    tighter = dict(prm, reach=2)
    rejects("R2", rec, spec, tighter)                      # routing rule differs
    rejects("R1", rec, spec, tighter, check_rule=False)    # hop longer than reach


def test_broken_hop_chain(ag, relayed):
    rec, prm, spec = ag
    pi, qi, ri = relayed
    bad = deep(rec)
    bad["phases"][pi]["rounds"][qi]["routes"][ri]["hops"][1][0] += 1
    rejects("R1", bad, spec, prm, check_rule=False)


def test_longer_arc_rejected():
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1], [3], tp)
    rec, prm = rec_and_params(op, np.array([100]), tp)
    spec = spec_alltoall([1], [3], [100])
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["routes"][0]["hops"] = [[1, 3, -1, 10, 1]]
    rejects("R2", bad, spec, prm)


# ---------------------------------------------------------------- R3
def test_segment_label_conflict():
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1, 2], [4, 5], tp)          # overlapping clockwise arcs, one round
    rec, prm = rec_and_params(op, np.array([100, 100]), tp)
    spec = spec_alltoall([1, 2], [4, 5], [100, 100])
    verify(rec, spec, prm)
    rnd = rec["phases"][0]["rounds"][0]
    assert len(rnd["routes"]) == 2
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["routes"][1]["hops"][0][4] = rnd["routes"][0]["hops"][0][4]
    rejects("R3", bad, spec, prm)


def test_per_label_tx_shared_across_directions():
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1, 1], [2, 12], tp)         # opposite directions from the same node
    rec, prm = rec_and_params(op, np.array([100, 100]), tp)
    spec = spec_alltoall([1, 1], [2, 12], [100, 100])
    verify(rec, spec, prm)
    bad = deep(rec)
    for r in bad["phases"][0]["rounds"][0]["routes"]:
        r["hops"][0][4] = 1
    rejects("R3", bad, spec, prm)


def test_aggregate_budget_exceeded(ag):
    rec, prm, spec = ag
    rejects("R3", rec, spec, dict(prm, w_tx=1, w_rx=1))


def test_peer_cap_exceeded():
    tp = stressed(N=16, lam=8, w=8, reach=0, d_peer=0, kappa=1000)
    op = osm([1, 1, 1], [2, 3, 4], tp)     # three peers of node 1 in one round
    rec, prm = rec_and_params(op, np.array([10, 10, 10]), tp)
    spec = spec_alltoall([1, 1, 1], [2, 3, 4], [10, 10, 10])
    verify(rec, spec, prm)
    assert len(rec["phases"][0]["rounds"]) == 1
    rejects("R3", rec, spec, dict(prm, d_peer=2))


def test_merged_rounds_conflict(ag):
    rec, prm, spec = ag
    bad = strip_claims(rec)
    ph = next(p for p in bad["phases"] if len(p["rounds"]) >= 2)
    ph["rounds"][0]["routes"] += ph["rounds"][1]["routes"]
    del ph["rounds"][1]
    rejects("R3", bad, spec, prm)


# ---------------------------------------------------------------- R4
def test_first_fit_nonconformance():
    """Physically feasible schedules that do not follow the deterministic rule."""
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1, 2], [2, 3], tp)
    rec, prm = rec_and_params(op, np.array([10, 10]), tp)
    spec = spec_alltoall([1, 2], [2, 3], [10, 10])
    verify(rec, spec, prm)
    assert len(rec["phases"][0]["rounds"]) == 1
    bad = strip_claims(rec)                     # a higher label than first-fit assigns
    bad["phases"][0]["rounds"][0]["routes"][0]["hops"][0][4] = 4
    verify(bad, spec, prm, check_packing=False)
    rejects("R4", bad, spec, prm, check_packing=True)
    bad = strip_claims(rec)                     # a route deferred without need
    moved = bad["phases"][0]["rounds"][0]["routes"].pop()
    bad["phases"][0]["rounds"].append(dict(routes=[moved]))
    verify(bad, spec, prm, check_packing=False)
    rejects("R4", bad, spec, prm, check_packing=True)


def test_dissemination_rule_violations():
    tp = stressed(N=24, lam=2, w=2, reach=3, d_peer=3, kappa=1000)
    op = [o for o in allgather_ops(3, 12, tp) if o.candidate.startswith("E-A")][0]
    rec, prm = rec_and_params(op, 100, tp)
    spec = spec_allgather(range(1, 4), range(1, 13), 100)
    verify(rec, spec, prm)
    ph = next(p for p in rec["phases"] if p["rule"] == "dissemination")
    bad = deep(rec)
    bph = next(p for p in bad["phases"] if p["rule"] == "dissemination")
    bph["targets"] = list(reversed(bph["targets"]))
    rejects("R4", bad, spec, prm, check_packing=True)
    # a transfer with a different (still holding, still in-reach) sender
    bad = deep(rec)
    bph = next(p for p in bad["phases"] if p["rule"] == "dissemination")
    r = bph["rounds"][0]["routes"][0]
    alt = r["src"] % 3 + 1
    r["src"] = alt
    rejects("R4", bad, spec, prm, check_packing=True)


# ---------------------------------------------------------------- R5
def test_chunk_forwarded_before_reception(ag, relayed):
    rec, prm, spec = ag
    pi, qi, ri = relayed
    bad = deep(rec)
    r = bad["phases"][pi]["rounds"][qi]["routes"][ri]
    for t in r["tx"]:
        if t[0] == 2 and t[1] == 1:
            t[2] = 1          # hop 2 carries chunk 1 in step 1 (same as hop 1)
    rejects("R5", bad, spec, prm)


def test_step_rule_violated_but_causal(ag, relayed):
    rec, prm, spec = ag
    pi, qi, ri = relayed
    bad = deep(rec)
    r = bad["phases"][pi]["rounds"][qi]["routes"][ri]
    for t in r["tx"]:
        if t[0] == len(r["hops"]):
            t[2] += 1         # delay the last hop by one step
    rejects("R5", bad, spec, prm)


def test_tail_rounded_up(ag, relayed):
    rec, prm, spec = ag
    pi, qi, ri = relayed
    bad = deep(rec)
    r = bad["phases"][pi]["rounds"][qi]["routes"][ri]
    assert r["bytes"] % prm["kappa_bytes"] != 0
    for t in r["tx"]:
        t[3] = prm["kappa_bytes"]
    rejects("R5", bad, spec, prm)


def test_direct_only_round_split():
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1], [2], tp)
    rec, prm = rec_and_params(op, np.array([2500]), tp)
    spec = spec_alltoall([1], [2], [2500])
    verify(rec, spec, prm)
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["routes"][0]["tx"] = [[1, 1, 1, 1000], [1, 2, 2, 1000], [1, 3, 3, 500]]
    rejects("R5", bad, spec, prm)


def test_circuit_reuse_in_same_step(ag, relayed):
    rec, prm, spec = ag
    pi, qi, ri = relayed
    bad = deep(rec)
    r = bad["phases"][pi]["rounds"][qi]["routes"][ri]
    for t in r["tx"]:
        if t[0] == 1 and t[1] == 2:
            t[2] = 1          # chunks 1 and 2 on the first hop in step 1
    rejects("R5", bad, spec, prm)


# ---------------------------------------------------------------- R6
def test_item_not_held(ag):
    rec, prm, spec = ag
    bad = deep(rec)
    r = bad["phases"][0]["rounds"][0]["routes"][0]
    r["item"] = 12 if r["item"] != 12 else 11
    rejects("R6", bad, spec, prm)


def test_item_forwarded_in_round_of_receipt(ag):
    """A node receives shard s in round k and sends it in the same round."""
    rec, prm, spec = ag
    bad = strip_claims(rec)
    ph0 = bad["phases"][0]
    r = ph0["rounds"][0]["routes"][0]
    assert all(len(x["hops"]) == 1 for x in ph0["rounds"][0]["routes"])   # direct-only round
    fwd = dict(src=r["dst"], dst=r["dst"] + 12, item=r["item"], bytes=r["bytes"],
               hops=[[r["dst"], r["dst"] + 12, 1, 12, 64]])
    ph0["rounds"][0]["routes"].append(fwd)
    rejects("R6", bad, spec, dict(prm, w_tx=64, w_rx=64, lam=64, d_peer=0, reach=12))


def test_missing_delivery(ag):
    rec, prm, spec = ag
    bad = deep(rec)
    last = bad["phases"][-1]["rounds"][-1]["routes"]
    last.pop()
    if not last:
        bad["phases"][-1]["rounds"].pop()
    rejects("R6", bad, spec, prm)


def test_duplicate_delivery(ag):
    rec, prm, spec = ag
    bad = strip_claims(rec)
    r = deep(bad["phases"][-1]["rounds"][-1]["routes"][-1])
    bad["phases"][-1]["rounds"].append(dict(routes=[r]))
    rejects("R6", bad, spec, prm)


def test_alltoall_wrong_blocks():
    tp = stressed(N=12, lam=4, w=4, reach=0, d_peer=0, kappa=1000)
    op = osm([1, 2], [3, 4], tp)
    rec, prm = rec_and_params(op, np.array([100, 200]), tp)
    spec = spec_alltoall([1, 2], [3, 4], [100, 200])
    verify(rec, spec, prm)
    rejects("R6", rec, spec_alltoall([1, 2], [3, 4], [100, 201]), prm)       # wrong bytes
    rejects("R6", rec, spec_alltoall([1], [3], [100]), prm)                  # unrequested block
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["routes"][0]["dst"] = 1
    rejects("R6", bad, spec, prm, check_rule=False)                          # self-transfer


# ---------------------------------------------------------------- R7
@pytest.fixture(scope="module")
def ar():
    tp = stressed(N=16, lam=2, w=2, reach=3, d_peer=3, kappa=1000)
    op = wrht_op(list(range(1, 10)), 3, "collect", tp)   # groups [1-3],[4-6],[7-9]; reps 2,5,8
    V = 1500
    rec, prm = rec_and_params(op, V, tp)
    spec = spec_allreduce(range(1, 10), V)
    verify(rec, spec, prm)
    return rec, prm, spec


def test_double_counted_contribution(ar):
    rec, prm, spec = ar
    bad = strip_claims(rec)
    top = next(p for p in bad["phases"] if p["name"] == "collect_top")
    route = dict(src=1, dst=5, item=1, bytes=1500,
                 hops=[[1, 4, 1, 3, 1], [4, 5, 1, 1, 1]])
    top["rounds"].append(dict(routes=[route]))
    top["reduce_counts"][5] = top["reduce_counts"].get(5, 0) + 1
    rejects("R7", bad, spec, prm)


def test_wrong_declared_reduction_count(ar):
    rec, prm, spec = ar
    bad = deep(rec)
    ph = bad["phases"][0]
    k = next(iter(ph["reduce_counts"]))
    ph["reduce_counts"][k] += 1
    rejects("R7", bad, spec, prm)


def test_missing_distribution(ar):
    rec, prm, spec = ar
    bad = strip_claims(rec)
    last = bad["phases"][-1]
    last["rounds"][-1]["routes"].pop()
    if not last["rounds"][-1]["routes"]:
        last["rounds"].pop()
    rejects("R7", bad, spec, prm)


# ---------------------------------------------------------------- R8
def _two_phase_record(second_routes, setups):
    prm = dict(N=8, lam=2, w_tx=2, w_rx=2, reach=0, d_peer=0, kappa_bytes=1000, rate_bps=25e9,
               eps=1.0, t_tx=1e-8, t_rx=1e-8, t_fwd=0.0, seg_len_m=1.0, v_g=2e8, t_setup=25e-6,
               beta_red=366e9)
    r1 = dict(src=1, dst=2, item=1, bytes=10, hops=[[1, 2, 1, 1, 1]])
    rec = dict(op="A2A", phases=[
        dict(name="a", semantics="transfer", rounds=[dict(routes=[r1], setup=setups[0])]),
        dict(name="b", semantics="transfer", rounds=[dict(routes=second_routes, setup=setups[1])])])
    return rec, prm


def test_setup_reuse_subset_allowed_and_misclaims_rejected():
    r2 = dict(src=1, dst=2, item=2, bytes=10, hops=[[1, 2, 1, 1, 1]])
    spec = dict(required={(1, 2): 20})
    rec, prm = _two_phase_record([r2], [1, 0])
    out = verify(rec, spec, prm, check_packing=False)
    assert out["setups"] == 1
    rec, prm = _two_phase_record([r2], [1, 1])                  # reuse not credited
    rejects("R8", rec, spec, prm)
    r3 = dict(src=2, dst=3, item=3, bytes=10, hops=[[2, 3, 1, 1, 1]])
    spec3 = dict(required={(1, 2): 20, (2, 3): 10})
    rec, prm = _two_phase_record([r2, r3], [1, 0])              # superset needs setup
    rejects("R8", rec, spec3, prm)
    r2b = dict(src=1, dst=2, item=2, bytes=10, hops=[[1, 2, 1, 1, 2]])
    rec, prm = _two_phase_record([r2b], [1, 0])                 # different label
    rejects("R8", rec, spec, prm)


def test_setup_wrongly_skipped(ag):
    rec, prm, spec = ag
    bad = deep(rec)
    for ph in bad["phases"]:
        for rnd in ph["rounds"]:
            if rnd["setup"] == 1 and ph is not bad["phases"][0]:
                rnd["setup"] = 0
                rejects("R8", bad, spec, prm)
                return
    pytest.fail("no later round with setup")


# ---------------------------------------------------------------- R9 / R10
def test_wrong_timing_claims(ag):
    rec, prm, spec = ag
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["body"] *= 1.001
    rejects("R9", bad, spec, prm)
    bad = deep(rec)
    bad["phases"][0]["rounds"][0]["steps"] += 1
    rejects("R9", bad, spec, prm)
    bad = deep(rec)
    bad["claimed"]["transport"] *= 0.999
    rejects("R9", bad, spec, prm)


def test_understated_buffers(ag, ar):
    rec, prm, spec = ag
    bad = deep(rec)
    node = next(iter(bad["claimed"]["relay_peak"]))
    bad["claimed"]["relay_peak"][node] *= 0.5
    rejects("R10", bad, spec, prm)
    bad = deep(rec)
    node = next(iter(bad["claimed"]["gathered"]))
    bad["claimed"]["gathered"][node] -= 1
    rejects("R10", bad, spec, prm)
    rec, prm, spec = ar
    bad = deep(rec)
    node = next(iter(bad["claimed"]["recv_peak"]))
    bad["claimed"]["recv_peak"][node] -= 1
    rejects("R10", bad, spec, prm)
    bad = deep(rec)
    node = next(iter(bad["claimed"]["accumulators"]))
    bad["claimed"]["accumulators"][node] = 0
    rejects("R10", bad, spec, prm)
    bad = deep(rec)
    bad["claimed"]["reduction"] *= 0.5
    rejects("R9", bad, spec, prm)
