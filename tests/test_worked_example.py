"""Reproduce the author-supplied N=4 worked example (handoff worked_example_values.txt)
with the general constructors. Values in milliseconds."""
import pytest

from lhas.collectives.families import wrht_ops, allgather_ops
from lhas.transport.schedule import evaluate
from helpers import TP4, BETA

EXPECTED = {  # from verify_worked_example.py (approved handoff)
    "AGCONV": 4.1354479200, "ARCONV": 17.1686173084, "DPCONV": 0.0742303462, "MPCONV": 21.3040652284,
    "AGFC": 0.1089160800, "ARFC": 0.3748991287, "DPFC": 22.4166542396, "MPFC": 0.4838152087,
}
LAYERS = {"CONV": (64 * 64 * 3 * 3 * 4, 64 * 64 * 56 * 56 * 4), "FC": (4096 * 4096 * 4, 64 * 4096 * 4)}


@pytest.mark.parametrize("name", ["CONV", "FC"])
def test_worked_example(name):
    W, X = LAYERS[name]
    ar = wrht_ops([1, 2, 3, 4], TP4)
    ag = allgather_ops(4, 4, TP4)
    arx = min((evaluate(o, X, TP4, BETA).latency, o.candidate) for o in ar)
    arw = min((evaluate(o, W, TP4, BETA).latency, o.candidate) for o in ar)
    agx = min((evaluate(o, X // 4, TP4, BETA).latency, o.candidate) for o in ag)
    got = {"AG": agx[0], "AR": arx[0], "DP": arw[0], "MP": agx[0] + arx[0]}
    for k, v in got.items():
        assert v * 1e3 == pytest.approx(EXPECTED[k + name], rel=0, abs=5e-10)
    assert arx[1] == "WRHT|m=4|exchange" and agx[1].startswith("EQ|k=1")
