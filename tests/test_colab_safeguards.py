"""The Colab notebook's safeguards and the ingestion checks (D17), exercised on CPU
by colab/cpu_smoke_test.py (no GPU timing; outputs are not measurements)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_and_ingestion_safeguards(tmp_path):
    pytest.importorskip("torchvision")
    sys.path.insert(0, str(ROOT / "colab"))
    import cpu_smoke_test
    res = cpu_smoke_test.run(tmp_path, verbose=False)
    assert res["measured"] == res["shapes_run"] and res["table_entries"] >= res["measured"]
    assert all(res["rejections"].values()) and res["resume_guard"]
    assert {"nan_timing", "missing_column", "duplicate_signature", "no_fingerprint"} <= set(res["rejections"])


def test_stored_t4_zip_ingests_to_the_stored_table():
    """Hardened ingestion still reproduces the table used by the profiled runs exactly."""
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    import ingest_colab as IC
    t = IC.ingest(str(ROOT / "results/colab/Tesla_T4_2026-09-28/lhas_profile_Tesla_T4.zip"), 1024)
    assert t == json.loads((ROOT / "configs/profiles/profiled_Tesla_T4_B1024.json").read_text())
