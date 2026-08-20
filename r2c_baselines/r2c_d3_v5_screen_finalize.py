from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import r2c_d3_v5_screen_queue as screen
from .utils import atomic_json, utc_now

FROZEN_RANKER = screen._rank_candidates


def rank_with_overall(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen Phase A ordering with its serialization-only ordinal."""
    ranked = FROZEN_RANKER(frame).copy()
    ranked["overall_rank"] = range(1, len(ranked) + 1)
    return ranked


def finalize() -> dict[str, Any]:
    manifest = json.loads(screen.MANIFEST_PATH.read_text(encoding="utf-8"))
    screen._assert_frozen_manifest(manifest)
    if any(job.get("status") != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot recover selection before all frozen screen jobs complete")

    if screen.SELECTION_PATH.exists():
        selection = json.loads(screen.SELECTION_PATH.read_text(encoding="utf-8"))
    else:
        original_ranker = screen._rank_candidates
        screen._rank_candidates = rank_with_overall
        try:
            selection = screen.freeze_selection(manifest)
        finally:
            screen._rank_candidates = original_ranker

    state = json.loads(screen.STATE_PATH.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "screen_completed_selection_frozen",
            "current_job_id": None,
            "updated_utc": utc_now(),
            "completed": len(manifest["jobs"]),
            "failed": 0,
            "total": len(manifest["jobs"]),
            "selection_recovery": "postprocessing_overall_rank_serialization_fix",
            "selected_candidates": [
                {"alpha": value["alpha"], "beta": value["beta"]}
                for value in selection["selected_candidates"]
            ],
        }
    )
    atomic_json(screen.STATE_PATH, state)
    return state


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
