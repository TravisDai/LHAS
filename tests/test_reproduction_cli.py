"""Guard the CI comparison against silent omissions and provenance-only failures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_reproduction", ROOT / "scripts/check_reproduction.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    original = json.loads((ROOT / "results/raw/final/alexnet_N64.json").read_text())
    directory = tmp_path / "results/raw/final"
    directory.mkdir(parents=True)
    (directory / "alexnet_N64.json").write_text(json.dumps(original))
    monkeypatch.setattr(CHECK, "ROOT", tmp_path)

    def run(mutator=lambda d: None, *args):
        fresh = deepcopy(original)
        mutator(fresh)
        path = tmp_path / "new.json"
        path.write_text(json.dumps(fresh))
        monkeypatch.setattr(sys, "argv", ["check_reproduction.py", str(path), *args])
        return CHECK.main()

    return run


def test_strict_source_hash_is_still_required(comparison):
    assert comparison(lambda d: d["provenance"].update(source_sha256="different")) == 1


def test_ci_source_exception_is_explicit_and_bounded(comparison, capsys):
    assert comparison(lambda d: d["provenance"].update(source_sha256="different"),
                      "--allow-source-change") == 0
    assert "SKIP provenance.source_sha256 equality" in capsys.readouterr().out
    assert comparison(lambda d: d["provenance"].update(nominal_spec_sha256="different"),
                      "--allow-source-change") == 1


@pytest.mark.parametrize("baseline", ["dp_N", "dp_best", "owt"])
def test_required_baseline_cannot_disappear(comparison, baseline):
    assert comparison(lambda d: d["baselines"].pop(baseline),
                      "--require-baselines", "dp_N,dp_best,owt",
                      "--allow-source-change") == 1


def test_changed_cost_still_fails_ci(comparison):
    assert comparison(lambda d: d["baselines"]["owt"].update(cost=1),
                      "--allow-source-change", "--require-baselines", "dp_N,dp_best,owt") == 1


def test_requested_subset_agrees(comparison):
    def subset(d):
        d["baselines"] = {k: d["baselines"][k] for k in ("dp_N", "dp_best", "owt")}
    assert comparison(subset, "--require-baselines", "dp_N,dp_best,owt") == 0
