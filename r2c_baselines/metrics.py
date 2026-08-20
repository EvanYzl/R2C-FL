from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def participation_jfi(counts: Iterable[int | float]) -> float:
    x = np.asarray(list(counts), dtype=np.float64)
    denom = float(len(x) * np.square(x).sum())
    return float(np.square(x.sum()) / denom) if denom > 0 else 0.0


def worst10_participation(counts: Iterable[int | float]) -> float:
    x = np.sort(np.asarray(list(counts), dtype=np.float64))
    n = max(1, int(math.ceil(0.1 * len(x))))
    return float(x[:n].mean())


def recovery_auc20(
    rounds: Iterable[int], accuracies: Iterable[float], event_round: int | None
) -> dict[str, object]:
    """Frozen recovery-deficit AUC over exactly 20 pre/post completed rounds."""
    if event_round is None:
        return {
            "pre_event_accuracy": None,
            "max_drop": None,
            "recovery_deficit_auc20": None,
            "recovery_auc20_complete": False,
            "recovery_missing_reason": "scenario_has_no_registered_event",
            "recovery_half_life_rounds": None,
            "post_event_round20_accuracy": None,
        }
    values = {int(r): float(a) for r, a in zip(rounds, accuracies)}
    pre_rounds = list(range(int(event_round) - 20, int(event_round)))
    post_rounds = list(range(int(event_round) + 1, int(event_round) + 21))
    missing_pre = [r for r in pre_rounds if r not in values]
    missing_post = [r for r in post_rounds if r not in values]
    if missing_pre or missing_post:
        reasons = []
        if missing_pre:
            reasons.append(f"missing_pre_rounds:{','.join(map(str, missing_pre))}")
        if missing_post:
            reasons.append(f"missing_post_rounds:{','.join(map(str, missing_post))}")
        return {
            "pre_event_accuracy": None,
            "max_drop": None,
            "recovery_deficit_auc20": None,
            "recovery_auc20_complete": False,
            "recovery_missing_reason": ";".join(reasons),
            "recovery_half_life_rounds": None,
            "post_event_round20_accuracy": None,
        }
    pre = float(np.mean([values[r] for r in pre_rounds]))
    post = np.asarray([values[r] for r in post_rounds], dtype=np.float64)
    deficits = np.maximum(0.0, pre - post)
    first_drop = max(0.0, pre - float(post[0]))
    half_life = None
    if first_drop == 0.0:
        half_life = 0
    else:
        threshold = first_drop / 2.0
        for h in range(5, 21):
            trailing = float(post[h - 5 : h].mean())
            if max(0.0, pre - trailing) <= threshold:
                half_life = h
                break
    return {
        "pre_event_accuracy": pre,
        "max_drop": float(np.max(np.maximum(0.0, pre - post))),
        "recovery_deficit_auc20": float(deficits.mean()),
        "recovery_auc20_complete": True,
        "recovery_missing_reason": None,
        "recovery_half_life_rounds": half_life,
        "post_event_round20_accuracy": float(post[-1]),
    }


def event_window_role(round_number: int, event_round: int | None) -> tuple[str | None, int | None]:
    if event_round is None:
        return None, None
    offset = int(round_number) - int(event_round)
    if -20 <= offset <= -1:
        return "pre20", offset
    if offset == 0:
        return "event", offset
    if 1 <= offset <= 20:
        return "post20", offset
    return "outside", offset

