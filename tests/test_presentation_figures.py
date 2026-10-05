"""The paper's evaluation figures 6-10 and 12 are drawn from a snapshot of the frozen raw results
(results/processed/presentation_plot_data.json). Re-extracting the snapshot from results/raw must
reproduce the stored snapshot exactly, and every recorded raw-file hash must still match."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "results" / "processed" / "presentation_plot_data.json"
FIGURES = ["fig_main_comparison_compact", "fig_ablations_families", "fig_sensitivity_compact",
           "fig_batch_comparison", "fig_layer_configurations", "fig_profiled_comparison"]


def test_recorded_raw_hashes_match():
    d = json.loads(SNAPSHOT.read_text())
    assert len(d["source_sha256"]) == 146
    for name, expected in d["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_snapshot_regenerates_from_raw_results(tmp_path):
    pytest.importorskip("matplotlib")
    # a separate process keeps the plotting script's matplotlib settings out of the test session
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_presentation_figures.py"),
                    "--output", str(tmp_path)], check=True, cwd=ROOT, capture_output=True)
    assert json.loads((tmp_path / "presentation_plot_data.json").read_text()) == json.loads(SNAPSHOT.read_text())
    for name in FIGURES:
        for ext in ("pdf", "png", "svg"):
            assert (tmp_path / f"{name}.{ext}").stat().st_size > 0
