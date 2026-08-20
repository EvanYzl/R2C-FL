from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from r2c_baselines import r2c_d3_v14_validation_screen_erratum_queue as queue
from r2c_baselines.utils import config_hash, sha256_file


def _load_erratum_auditor() -> ModuleType:
    path = Path(__file__).with_name("audit_r2c_v14_run_erratum.py")
    spec = importlib.util.spec_from_file_location("test_v14_erratum_auditor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_window_frame() -> pd.DataFrame:
    offsets = list(range(-20, 0)) + [0] + list(range(1, 21))
    roles = ["pre20"] * 20 + ["event"] + ["post20"] * 20
    return pd.DataFrame(
        {
            "round": list(range(480, 521)),
            "event_offset_round": offsets,
            "auc20_window_role": roles,
            "sentinel": [f"row-{index}" for index in range(41)],
        }
    )


@pytest.fixture(scope="module")
def prospective_manifest() -> dict[str, Any]:
    paths = (
        queue.MANIFEST_PATH,
        queue.STATE_PATH,
        queue.EVENTS_PATH,
        queue.RESULT_PATH,
        queue.RUNS_PATH,
        queue.RUNS_CSV_PATH,
    )
    before = {
        str(path): (path.exists(), sha256_file(path) if path.is_file() else None)
        for path in paths
    }
    manifest = queue.build_manifest(persist=False)
    after = {
        str(path): (path.exists(), sha256_file(path) if path.is_file() else None)
        for path in paths
    }
    assert after == before
    return manifest


def test_compatibility_view_maps_only_canonical_roles() -> None:
    auditor = _load_erratum_auditor()
    recorded = _exact_window_frame()
    original = recorded.copy(deep=True)
    compatible = auditor.compatibility_round_frame(recorded)

    pd.testing.assert_frame_equal(recorded, original)
    assert compatible is not recorded
    assert compatible["sentinel"].tolist() == recorded["sentinel"].tolist()
    assert compatible["event_offset_round"].tolist() == recorded[
        "event_offset_round"
    ].tolist()
    assert compatible["auc20_window_role"].value_counts().to_dict() == {
        "before": 20,
        "after": 20,
        "event": 1,
    }


def test_compatibility_view_rejects_recorded_auditor_aliases() -> None:
    auditor = _load_erratum_auditor()
    recorded = _exact_window_frame()
    recorded.loc[0, "auc20_window_role"] = "before"
    with pytest.raises(AssertionError, match="auditor-only"):
        auditor.compatibility_round_frame(recorded)


def test_prospective_manifest_preserves_original_grid_and_sealing(
    prospective_manifest: dict[str, Any],
) -> None:
    manifest = prospective_manifest
    assert manifest["recovery_authority_id"] == queue.RECOVERY_AUTHORITY_ID
    assert manifest["recovery_mode"] == queue.RECOVERY_MODE
    assert manifest["candidate_order"] == list(queue.base.CANDIDATES)
    assert manifest["job_order"] == [job["job_id"] for job in manifest["jobs"]]
    assert [job["status"] for job in manifest["jobs"]] == [
        "completed",
        "pending",
        "pending",
    ]
    assert [job["execution_required"] for job in manifest["jobs"]] == [
        False,
        True,
        True,
    ]
    assert manifest["jobs"][0]["actual_run_id"] == queue.PRESERVED_RUN_ID
    assert manifest["execution_plan"]["new_execution_count"] == 2
    assert manifest["performance_sealed_until_terminal"] is True
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    assert manifest["preserved_run_preflight"] == {
        "status": "passed",
        "run_id": queue.PRESERVED_RUN_ID,
        "audit_erratum_id": queue.ERRATUM_ID,
        "round_rows": 1000,
        "before_count": 20,
        "after_count": 20,
        "recorded_tables_mutated": False,
    }


def test_prospective_manifest_has_exact_predecessor_and_frozen_hash(
    prospective_manifest: dict[str, Any],
) -> None:
    manifest = prospective_manifest
    assert (
        manifest["superseded_authority"]["immutable_manifest_sha256"]
        == queue.ORIGINAL_IMMUTABLE_MANIFEST_SHA256
    )
    assert (
        sha256_file(queue.ORIGINAL_IMMUTABLE_MANIFEST_PATH)
        == queue.ORIGINAL_IMMUTABLE_MANIFEST_SHA256
    )
    assert manifest["audit_erratum"] == queue._erratum_descriptor()
    assert manifest["preserved_run_lineage"] == queue._preserved_run_lineage()
    assert manifest["frozen_spec_hash"] == config_hash(queue._frozen_spec(manifest))
    queue._assert_frozen_manifest(manifest)


def test_erratum_execution_fields_are_immutable(
    prospective_manifest: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(prospective_manifest)
    tampered["jobs"][1]["execution_required"] = False
    with pytest.raises(RuntimeError):
        queue._assert_frozen_manifest(tampered)


def test_erratum_does_not_change_original_authority_hashes() -> None:
    assert (
        sha256_file(queue.ORIGINAL_IMMUTABLE_MANIFEST_PATH)
        == queue.ORIGINAL_IMMUTABLE_MANIFEST_SHA256
    )
    assert (
        sha256_file(queue.ERRATUM_AUDITOR_PATH.with_name("audit_r2c_v14_run.py"))
        == queue.FROZEN_AUDITOR_SHA256
    )
