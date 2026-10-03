"""Profiled input mode: measured local times replace analytic T_cp only where
measured; missing configurations raise instead of being extrapolated, and tables
measured for another batch, manifest, precision or local shape are rejected (D17)."""
import copy

import pytest

from lhas.params import load_approved
from lhas.collectives.service import CollectiveService
from lhas.model.chain import ChainModel, Settings, DP, MP
from lhas.model.workload import synthetic_chain, load_workload
from lhas.model.profile import ProfileMismatch, expected_local_shape

SPEC = [dict(kind="fc", n_in=8, n_out=4, x_shape=(8,), y_shape=(4,)),
        dict(kind="fc", n_in=4, n_out=4, x_shape=(4,), y_shape=(4,))]


def _table(wl, B=4):
    return {"input_mode": "profiled_TestGPU", "B": B, "precision": "float32", "tf32": False,
            "manifest_sha256": {wl.name: wl.manifest_sha256},
            "table": {"toy|toy_0|DP|2": dict(forward=1.0, input_grad=0.5, weight_grad=0.25, batch_local=2,
                                             n_out_local=4),
                      "toy|toy_1|MP|4": dict(forward=2.0, input_grad=1.0, weight_grad=0.5, batch_local=4,
                                             n_out_local=1)}}


def _model(wl, table):
    tp, cp = load_approved(4)
    return ChainModel(wl, CollectiveService(tp, cp), Settings(B=4, raw_input_grad="omit"), compute_table=table)


def _toy():
    wl = synthetic_chain("toy", SPEC)
    wl.manifest_sha256 = "toyhash"
    return wl


def test_profiled_table_used_and_missing_entries_refused():
    wl = _toy()
    m = _model(wl, _table(wl))
    assert m.tcp(0, (DP, 2)) == pytest.approx(1.25)       # raw-input gradient omitted for the first layer
    assert m.tcp(1, (MP, 4)) == pytest.approx(3.5)
    with pytest.raises(KeyError, match="unmeasured"):
        m.tcp(0, (DP, 4))


@pytest.mark.parametrize("edit", ["B", "manifest", "precision", "tf32", "dims"])
def test_profiled_table_mismatches_rejected(edit):
    wl = _toy()
    t = copy.deepcopy(_table(wl))
    if edit == "B":
        t["B"] = 8
    elif edit == "manifest":
        t["manifest_sha256"]["toy"] = "other"
    elif edit == "precision":
        t["precision"] = "float16"
    elif edit == "tf32":
        t["tf32"] = True
    else:
        t["table"]["toy|toy_0|DP|2"]["batch_local"] = 4    # measured at another local batch
    with pytest.raises(ProfileMismatch):
        _model(wl, t).tcp(0, (DP, 2))


def test_expected_local_shape_from_manifest():
    wl = load_workload("alexnet")
    L = wl.layers[0]
    assert len(wl.manifest_sha256) == 64
    d = expected_local_shape(wl, L, (DP, 8), 1024)
    assert d["batch_local"] == 128 and d["n_out_local"] == L.n_out and d["kernel"] == [11, 11]
    d = expected_local_shape(wl, L, (MP, 4), 1024)
    assert d["batch_local"] == 1024 and d["n_out_local"] == L.n_out // 4 and d["stride"] == [4, 4]
