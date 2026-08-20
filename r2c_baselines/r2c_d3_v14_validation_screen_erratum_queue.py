from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import r2c_d3_v14_validation_screen_queue as base
from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .utils import (
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


RECOVERY_AUTHORITY_ID = "R2C_V14_CMTR_M1_AUDIT_ERRATUM_CONTINUATION_20260819"
RECOVERY_MODE = "prospective_audit_erratum_continuation"
ORIGINAL_IMMUTABLE_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_d3_v14_validation_screen_manifest_20260819T142454.066621Z.json"
)
ORIGINAL_IMMUTABLE_MANIFEST_SHA256 = (
    "5656b90d9ad6b2e53e2ea795ad0e36e2121c7d5cf380e96b314bb7f523ba4ca3"
)
ORIGINAL_ACTIVE_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_d3_v14_validation_screen_manifest.json"
)
ORIGINAL_STATE_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_queue_state.json"
ORIGINAL_FAILED_AUDIT_LOG = (
    QUEUE_ROOT
    / "worker_logs"
    / "A-R2C-D3-S4-V14SCREEN-CMTR-B950-R20-s20260810.v14.audit.log"
)
PRESERVED_RUN_ID = "A-R2C-D3-S4-V14SCREEN-CMTR-B950-R20-s20260810"
PRESERVED_RUN_DIR = RUN_ROOT / PRESERVED_RUN_ID

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_queue_state.json"
EVENTS_PATH = (
    QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_scheduler_events.parquet"
)
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_erratum_runs.csv"
ERRATUM_AUDITOR_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_v14_run_erratum.py"
)
ERRATUM_ID = "R2C_V14_AUDIT_ROLE_ALIAS_ERRATUM_20260819"
FROZEN_AUDITOR_SHA256 = (
    "4e677fc85e00ad805a06dcd87d0c2133df3235f6e12ba085a86bcf0235c6f525"
)


_BASE_BUILD_MANIFEST = base.build_manifest
_BASE_FROZEN_SPEC = base._frozen_spec
_BASE_ASSERT_FROZEN_MANIFEST = base._assert_frozen_manifest
_BASE_IMPLEMENTATION_HASHES = base._implementation_hashes


def _terminal_predecessor_lineage() -> dict[str, Any]:
    required = (
        ORIGINAL_IMMUTABLE_MANIFEST_PATH,
        ORIGINAL_ACTIVE_MANIFEST_PATH,
        ORIGINAL_STATE_PATH,
        ORIGINAL_FAILED_AUDIT_LOG,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(ORIGINAL_IMMUTABLE_MANIFEST_PATH) != ORIGINAL_IMMUTABLE_MANIFEST_SHA256:
        raise RuntimeError("Original v14 immutable manifest hash drift")

    immutable = json.loads(
        ORIGINAL_IMMUTABLE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    active = json.loads(ORIGINAL_ACTIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(ORIGINAL_STATE_PATH.read_text(encoding="utf-8"))
    jobs = list(active.get("jobs", []))
    if (
        immutable.get("candidate_order") != list(base.CANDIDATES)
        or immutable.get("frozen_spec_hash")
        != config_hash(_BASE_FROZEN_SPEC(immutable))
        or state.get("status") != "failed"
        or int(state.get("completed", -1)) != 0
        or int(state.get("failed", -1)) != 1
        or int(state.get("total", -1)) != 3
        or bool(state.get("formal_test_access"))
        or bool(state.get("other_dataset_access"))
        or not bool(state.get("performance_sealed_until_terminal"))
        or len(jobs) != 3
        or jobs[0].get("status") != "failed"
        or int(jobs[0].get("attempts", -1)) != 1
        or jobs[0].get("actual_run_id") != PRESERVED_RUN_ID
        or [job.get("status") for job in jobs[1:]] != ["pending", "pending"]
    ):
        raise RuntimeError("Original v14 terminal-failure contract drift")

    return {
        "immutable_manifest_path": str(
            ORIGINAL_IMMUTABLE_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "immutable_manifest_sha256": ORIGINAL_IMMUTABLE_MANIFEST_SHA256,
        "active_manifest_path": str(
            ORIGINAL_ACTIVE_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "active_manifest_sha256": sha256_file(ORIGINAL_ACTIVE_MANIFEST_PATH),
        "terminal_state_path": str(ORIGINAL_STATE_PATH.relative_to(PROJECT_ROOT)),
        "terminal_state_sha256": sha256_file(ORIGINAL_STATE_PATH),
        "failed_audit_log_path": str(
            ORIGINAL_FAILED_AUDIT_LOG.relative_to(PROJECT_ROOT)
        ),
        "failed_audit_log_sha256": sha256_file(ORIGINAL_FAILED_AUDIT_LOG),
        "terminal_status": "failed",
        "terminal_completed": 0,
        "terminal_failed": 1,
        "failed_job_id": PRESERVED_RUN_ID,
        "failed_attempt": 1,
        "failure_class": "frozen_auditor_role_alias_contract_mismatch",
    }


def _preserved_run_lineage() -> dict[str, str]:
    required = [
        PRESERVED_RUN_DIR / "job.json",
        PRESERVED_RUN_DIR / "result.json",
        PRESERVED_RUN_DIR / "_SUCCESS.json",
        PRESERVED_RUN_DIR / "run_manifest.parquet",
        PRESERVED_RUN_DIR / "inputs" / "partition.json",
        PRESERVED_RUN_DIR / "inputs" / "trace.json",
    ]
    required.extend(sorted((PRESERVED_RUN_DIR / "tables").glob("*/_index.json")))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    immutable = json.loads(
        ORIGINAL_IMMUTABLE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    recorded_job = json.loads(
        (PRESERVED_RUN_DIR / "job.json").read_text(encoding="utf-8")
    )
    expected = base._job_spec(dict(immutable["jobs"][0]))
    if any(recorded_job.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Preserved v14 run job differs from immutable first job")

    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in required
    }


def _erratum_descriptor() -> dict[str, Any]:
    if not ERRATUM_AUDITOR_PATH.is_file():
        raise FileNotFoundError(ERRATUM_AUDITOR_PATH)
    frozen_auditor = ERRATUM_AUDITOR_PATH.with_name("audit_r2c_v14_run.py")
    if sha256_file(frozen_auditor) != FROZEN_AUDITOR_SHA256:
        raise RuntimeError("Frozen v14 auditor changed before erratum freeze")
    return {
        "erratum_id": ERRATUM_ID,
        "auditor_path": str(ERRATUM_AUDITOR_PATH.relative_to(PROJECT_ROOT)),
        "auditor_sha256": sha256_file(ERRATUM_AUDITOR_PATH),
        "frozen_auditor_path": str(frozen_auditor.relative_to(PROJECT_ROOT)),
        "frozen_auditor_sha256": FROZEN_AUDITOR_SHA256,
        "recorded_canonical_roles": {"before": "pre20", "after": "post20"},
        "frozen_auditor_aliases": {"before": "before", "after": "after"},
        "compatibility_view_only": True,
        "recorded_run_mutation_allowed": False,
        "applies_only_to": "v14 S4 runs with exact canonical pre20/post20 windows",
    }


def _preflight_preserved_run() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(ERRATUM_AUDITOR_PATH), str(PRESERVED_RUN_DIR)],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Preserved v14 run failed the prospective erratum auditor: "
            + process.stdout[-2000:]
        )
    payload = json.loads(process.stdout.strip().splitlines()[-1])
    if (
        payload.get("status") != "passed"
        or payload.get("run_id") != PRESERVED_RUN_ID
        or payload.get("audit_erratum_id") != ERRATUM_ID
        or bool(payload.get("recorded_tables_mutated"))
        or int(payload.get("round_rows", -1)) != base.ROUNDS
        or int(payload.get("canonical_window_contract", {}).get("before_count", -1))
        != 20
        or int(payload.get("canonical_window_contract", {}).get("after_count", -1))
        != 20
    ):
        raise RuntimeError("Prospective erratum audit payload contract drift")
    return {
        "status": "passed",
        "run_id": PRESERVED_RUN_ID,
        "audit_erratum_id": ERRATUM_ID,
        "round_rows": base.ROUNDS,
        "before_count": 20,
        "after_count": 20,
        "recorded_tables_mutated": False,
    }


def _implementation_hashes() -> dict[str, str]:
    values = dict(_BASE_IMPLEMENTATION_HASHES())
    package = Path(__file__).resolve().parent
    tests = Path(__file__).resolve().parents[1] / "tests"
    additions = (
        (Path(__file__).name, package / Path(__file__).name),
        (
            "tests/audit_r2c_v14_run_erratum.py",
            tests / "audit_r2c_v14_run_erratum.py",
        ),
        (
            "tests/test_r2c_d3_v14_validation_screen_erratum_queue.py",
            tests / "test_r2c_d3_v14_validation_screen_erratum_queue.py",
        ),
    )
    for name, path in additions:
        if not path.is_file():
            raise FileNotFoundError(path)
        values[name] = sha256_file(path)
    return values


def _frozen_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    value = _BASE_FROZEN_SPEC(manifest)
    for key in (
        "recovery_authority_id",
        "recovery_mode",
        "superseded_authority",
        "superseded_terminal_state",
        "audit_erratum",
        "preserved_run_lineage",
        "preserved_run_preflight",
        "execution_plan",
        "continuation_completion_rule",
    ):
        value[key] = manifest[key]
    return value


def _initial_job_state(job: dict[str, Any], position: int) -> None:
    job.update(
        {
            "continuation_authority_id": RECOVERY_AUTHORITY_ID,
            "execution_required": position != 0,
            "inherited_from_run_id": PRESERVED_RUN_ID if position == 0 else None,
            "inherited_via_erratum_id": ERRATUM_ID if position == 0 else None,
        }
    )
    if position == 0:
        job.update(
            {
                "status": "completed",
                "attempts": 0,
                "actual_run_id": PRESERVED_RUN_ID,
                "failure_reason": None,
            }
        )


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = ["completed", "pending", "pending"]
        if (
            [job.get("status") for job in existing.get("jobs", [])] != expected
            or any(int(job.get("attempts", 0)) for job in existing.get("jobs", []))
        ):
            raise RuntimeError("Refusing to rebuild a started v14 erratum continuation")

    predecessor = _terminal_predecessor_lineage()
    preserved_lineage = _preserved_run_lineage()
    erratum = _erratum_descriptor()
    preflight = _preflight_preserved_run()

    old_implementation = base._implementation_hashes
    old_frozen_spec = base._frozen_spec
    base._implementation_hashes = _implementation_hashes
    base._frozen_spec = _BASE_FROZEN_SPEC
    try:
        manifest = _BASE_BUILD_MANIFEST(persist=False)
    finally:
        base._implementation_hashes = old_implementation
        base._frozen_spec = old_frozen_spec

    manifest.update(
        {
            "schema_version": "1.1.0",
            "created_utc": utc_now(),
            "recovery_authority_id": RECOVERY_AUTHORITY_ID,
            "recovery_mode": RECOVERY_MODE,
            "superseded_authority": {
                "immutable_manifest_path": predecessor["immutable_manifest_path"],
                "immutable_manifest_sha256": predecessor[
                    "immutable_manifest_sha256"
                ],
            },
            "superseded_terminal_state": predecessor,
            "audit_erratum": erratum,
            "preserved_run_lineage": preserved_lineage,
            "preserved_run_preflight": preflight,
            "execution_plan": {
                "candidate_order_unchanged": True,
                "candidate_order": list(base.CANDIDATES),
                "inherited_completed_run_ids": [PRESERVED_RUN_ID],
                "new_execution_job_ids": [
                    job["job_id"] for job in manifest["jobs"][1:]
                ],
                "new_execution_count": 2,
                "selection_deferred_until_three_of_three_audited": True,
                "performance_read_before_terminal_allowed": False,
            },
            "continuation_completion_rule": (
                "the preserved first candidate must pass the hash-pinned erratum "
                "audit, the remaining two immutable candidates must each complete "
                "1000 rounds and pass the same erratum audit in original order, and "
                "no candidate performance may be read until all three are audited"
            ),
        }
    )
    for position, job in enumerate(manifest["jobs"]):
        _initial_job_state(job, position)
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))

    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = (
            QUEUE_ROOT
            / f"r2c_d3_v14_validation_screen_erratum_manifest_{stamp}.json"
        )
        atomic_json(immutable_path, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        now = utc_now()
        atomic_json(
            STATE_PATH,
            {
                "status": "ready",
                "created_utc": now,
                "updated_utc": now,
                "current_job_id": None,
                "completed": 1,
                "failed": 0,
                "total": 3,
                "inherited_completed": 1,
                "new_execution_remaining": 2,
                "all_runs_completed": False,
                "performance_sealed_until_terminal": True,
                "formal_test_access": False,
                "other_dataset_access": False,
                "recovery_authority_id": RECOVERY_AUTHORITY_ID,
                "frozen_spec_hash": manifest["frozen_spec_hash"],
                "immutable_manifest_path": str(immutable_path),
                "immutable_manifest_sha256": sha256_file(immutable_path),
                "superseded_immutable_manifest_sha256": (
                    ORIGINAL_IMMUTABLE_MANIFEST_SHA256
                ),
            },
        )
        atomic_parquet(
            EVENTS_PATH,
            pd.DataFrame(
                [
                    {
                        "schema_version": "2.1.0",
                        "job_id": PRESERVED_RUN_ID,
                        "run_id": PRESERVED_RUN_ID,
                        "event_utc": now,
                        "event_type": "inherited_completed_via_audit_erratum",
                        "attempt": 0,
                        "exit_code": 0,
                        "reason": ERRATUM_ID,
                    }
                ]
            ),
        )
    return manifest


def _audit_run(run_id: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [sys.executable, str(ERRATUM_AUDITOR_PATH), str(RUN_ROOT / run_id)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    old_implementation = base._implementation_hashes
    old_frozen_spec = base._frozen_spec
    base._implementation_hashes = _implementation_hashes
    base._frozen_spec = _frozen_spec
    try:
        _BASE_ASSERT_FROZEN_MANIFEST(manifest)
    finally:
        base._implementation_hashes = old_implementation
        base._frozen_spec = old_frozen_spec

    if (
        manifest.get("recovery_authority_id") != RECOVERY_AUTHORITY_ID
        or manifest.get("recovery_mode") != RECOVERY_MODE
        or manifest.get("superseded_terminal_state")
        != _terminal_predecessor_lineage()
        or manifest.get("audit_erratum") != _erratum_descriptor()
        or manifest.get("preserved_run_lineage") != _preserved_run_lineage()
        or manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest))
    ):
        raise RuntimeError("v14 erratum continuation evidence drift")
    jobs = list(manifest.get("jobs", []))
    expected_status = ["completed", "pending", "pending"]
    initial = all(int(job.get("attempts", 0)) == 0 for job in jobs)
    if initial and [job.get("status") for job in jobs] != expected_status:
        raise RuntimeError("v14 erratum continuation initial state drift")
    for position, job in enumerate(jobs):
        if (
            job.get("continuation_authority_id") != RECOVERY_AUTHORITY_ID
            or bool(job.get("execution_required")) is (position == 0)
            or job.get("inherited_from_run_id")
            != (PRESERVED_RUN_ID if position == 0 else None)
            or job.get("inherited_via_erratum_id")
            != (ERRATUM_ID if position == 0 else None)
        ):
            raise RuntimeError("v14 erratum per-job execution contract drift")


def _activate_base() -> None:
    base.MANIFEST_PATH = MANIFEST_PATH
    base.STATE_PATH = STATE_PATH
    base.EVENTS_PATH = EVENTS_PATH
    base.RESULT_PATH = RESULT_PATH
    base.RUNS_PATH = RUNS_PATH
    base.RUNS_CSV_PATH = RUNS_CSV_PATH
    base._implementation_hashes = _implementation_hashes
    base._frozen_spec = _frozen_spec
    base._assert_frozen_manifest = _assert_frozen_manifest
    base._audit_run = _audit_run


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    _activate_base()
    return base.worker()


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    sub.add_parser("worker")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        value = build_manifest(force=args.force)
    elif args.command == "worker":
        value = worker()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
