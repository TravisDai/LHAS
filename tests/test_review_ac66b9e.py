"""Regressions for findings A1, A3 and A4 of LHAS_ac66b9e_Independent_Review.md.
Each test reproduces the original defect's failure condition and asserts the corrected
behavior. (A2 narrows a documented guarantee; it is covered by the docstring check below.)"""
import copy
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "profiles" / "profiled_Tesla_T4_B1024.json"
sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------- A1
def _planner_models():
    """Profiled models of the four workloads under the nominal settings; their tcp() is the
    time the planner reads for a (layer, configuration)."""
    from lhas.params import load_approved
    from lhas.collectives.service import CollectiveService
    from lhas.model.workload import load_workload
    from lhas.model.chain import ChainModel, Settings
    from lhas.model.graph import build_structure
    from lhas.model.branch import GraphModel
    nominal = json.loads((ROOT / "configs" / "nominal_experiment.json").read_text())
    table = json.loads(PROFILE.read_text())
    tp, cp = load_approved(8)
    out = {}
    for w in ("alexnet", "vgg16"):
        m = ChainModel(load_workload(w), CollectiveService(tp, cp),
                       Settings(B=1024, raw_input_grad=nominal["raw_input_grad"]), compute_table=table)
        out[w] = (m, {L.name: i for i, L in enumerate(m.L)})
    for w in ("googlenet", "resnet50"):
        m = GraphModel(build_structure(load_workload(w)), CollectiveService(tp, cp, keep_structures=False),
                       Settings(B=1024, raw_input_grad=nominal["raw_input_grad"], norm_policy="sync"),
                       compute_table=table)
        out[w] = (m, {n: n for n in m.order})
    return out


def test_a1_configuration_times_are_the_planner_inputs():
    """Ranking and scaling use the time the nominal planner reads: the raw-input gradient of
    layers that consume the graph input is omitted (the original analysis added it)."""
    import analyse_profile as AP
    from lhas.model.chain import DP, MP
    rows = [r for r in AP.load(AP.ZIP)["rows"] if r["status"] == "measured"]
    tab = AP._config_times(rows)
    models = _planner_models()
    n, first_layer = 0, 0
    for (w, layer), d in tab.items():
        m, key = models[w]
        for (s, p), (t, _) in d.items():
            ref = m.tcp(key[layer], (DP if s == "DP" else MP, p))
            assert t == pytest.approx(ref, rel=1e-12, abs=0.0), (w, layer, s, p)
            n += 1
        L = m.L[key[layer]]
        first_layer += bool(L.input_is_graph_input)
    assert n == 2770 and first_layer == 4          # every measured entry; one raw-input layer per workload


def test_a1_heldout_validation_keeps_three_products():
    """The declared held-out protocol times all three products; A1 must not change it."""
    import analyse_profile as AP
    rows = [r for r in AP.load(AP.ZIP)["rows"] if r["status"] == "measured"]
    r = rows[0]
    assert AP.record_time(r) == sum(float(r[f"{k}_median_s"]) for k in AP.PARTS)
    groups = AP.group_by_signature(rows)
    v, _ = AP.grouped_validation(groups, 0)
    assert v["all"]["abs_rel_median"] == pytest.approx(0.591887, abs=5e-7)
    assert v["R"] == pytest.approx(2.913280e12, rel=1e-6)


# ---------------------------------------------------------------- A2
def test_a2_ingestion_documents_its_limited_fingerprint_guarantee():
    import ingest_colab as IC
    doc = IC.__doc__
    assert "does not independently authenticate" in doc
    assert "every resume of the run carried the same configuration fingerprint" not in doc


# ---------------------------------------------------------------- A3
META = ("environment.json", "config.json", "manifest_check.json", "shape_manifest.json")


def test_a3_notebook_writes_metadata_only_after_the_resume_guard():
    """Static check of the generated notebook (no torch needed): no metadata file is written
    before the resume-compatibility guard has passed."""
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "colab" / "LHAS_compute_profiling.ipynb"), as_version=4)
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    guard = src.index('if FP["fingerprint"] != FINGERPRINT')
    pre, post = src[:guard], src[guard:]
    for line in pre.splitlines():
        if "json.dump(" in line or '"w")' in line:
            assert not any(f in line for f in META), f"metadata written before the resume guard: {line.strip()}"
    assert "_write_meta(" not in pre
    for f in META:
        assert re.search(r'_write_meta\("' + re.escape(f), post), f


def test_a3_rejected_resume_modifies_nothing(tmp_path):
    """Behavioural check on CPU (colab/cpu_smoke_test.py): a compatible resume appends
    nothing and keeps the metadata; a resume with a changed configuration is rejected and
    leaves every existing file byte-identical."""
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(ROOT / "colab"))
    import cpu_smoke_test
    res = cpu_smoke_test.run(tmp_path, verbose=False)
    assert res["resume_guard"] and res["rejected_resume_unchanged"] and res["compatible_resume_unchanged"]


# ---------------------------------------------------------------- A4
def _pair(name="alexnet_N64.json"):
    import compare_reruns as CR
    old = json.loads((CR.OLD / "final" / name).read_text())
    new = json.loads((CR.RAW / "final" / name).read_text())
    return CR, old, new


def test_a4_changed_intermediate_incumbent_is_detected():
    CR, old, new = _pair()
    bad = copy.deepcopy(new)
    tr = bad["baselines"]["flexflow_mcmc"]["runs"][0]["trace"]
    tr[len(tr) // 2][3] *= 2.0                      # final best cost unchanged
    d, _ = CR.compare(old, bad)
    assert any("trace" in x for x in d)


def test_a4_missing_or_duplicate_seed_is_detected():
    CR, old, new = _pair()
    fewer = copy.deepcopy(new)
    fewer["baselines"]["flexflow_mcmc"]["runs"].pop()
    assert CR.compare(old, fewer)[0]
    dup = copy.deepcopy(new)
    runs = dup["baselines"]["flexflow_mcmc"]["runs"]
    runs[-1] = copy.deepcopy(runs[0])
    assert CR.compare(old, dup)[0]


def test_a4_elapsed_time_and_evaluation_counts_are_allowed_to_change():
    CR, old, new = _pair()
    ok = copy.deepcopy(new)
    for r in ok["baselines"]["flexflow_mcmc"]["runs"]:
        for x in r["trace"]:
            x[1] += 7
            x[2] *= 3.0
    assert CR.compare(old, ok)[0] == []


def test_a4_missing_required_file_gives_nonzero_exit(tmp_path, monkeypatch):
    import compare_reruns as CR
    raw = tmp_path / "raw"
    shutil.copytree(CR.OLD, raw / "archive_0892134")
    for tag, _ in CR.PAIRS:
        shutil.copytree(CR.RAW / tag, raw / tag)
    (raw / "final" / "vgg16_N256.json").unlink()
    out = tmp_path / "cmp.md"
    monkeypatch.setattr(CR, "RAW", raw)
    monkeypatch.setattr(CR, "OLD", raw / "archive_0892134")
    monkeypatch.setattr(CR, "OUT", out, raising=False)
    monkeypatch.setattr(CR, "ROOT", tmp_path)
    (tmp_path / "results" / "processed").mkdir(parents=True)
    assert CR.main() != 0
