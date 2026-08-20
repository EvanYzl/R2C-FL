from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CANDIDATE_M, NUM_CLIENTS, SELECTED_K


def _round_rng(seed: int, round_number: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(round_number), int(stream), 0x523243])
    )


def _weighted_choice_without_replacement(
    rng: np.random.Generator, ids: np.ndarray, size: int, weights: np.ndarray | None = None
) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    size = min(int(size), len(ids))
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if weights is None:
        return np.asarray(rng.choice(ids, size=size, replace=False), dtype=np.int64)
    probability = np.asarray(weights, dtype=np.float64)
    probability = np.maximum(probability, 0.0)
    if probability.sum() <= 0:
        probability = np.ones_like(probability)
    probability /= probability.sum()
    return np.asarray(rng.choice(ids, size=size, replace=False, p=probability), dtype=np.int64)


@dataclass
class Selection:
    admitted: np.ndarray
    selected: np.ndarray
    utility_scores: np.ndarray
    admission_prob: np.ndarray
    inclusion_prob: np.ndarray
    tier_ids: np.ndarray


class BaselineAdapter:
    def __init__(
        self,
        method_id: str,
        sample_counts: np.ndarray,
        seed: int,
        round_budget: int,
        config: dict[str, Any],
        selected_k: int = SELECTED_K,
        candidate_m: int = CANDIDATE_M,
    ) -> None:
        self.method_id = method_id
        self.sample_counts = np.asarray(sample_counts, dtype=np.float64)
        self.p = self.sample_counts / self.sample_counts.sum()
        self.seed = int(seed)
        self.round_budget = int(round_budget)
        self.config = dict(config)
        self.k = int(selected_k)
        self.m = int(candidate_m)
        self.utility = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
        self.tier_ids = np.full(NUM_CLIENTS, -1, dtype=np.int16)

        self.gap = np.ones(NUM_CLIENTS, dtype=np.int64)
        self.interval_history: list[list[int]] = [[] for _ in range(NUM_CLIENTS)]
        self.fedau_k = int(self.config.get("fedau_k", 50))

        self.f3ast_beta = float(self.config.get("f3ast_beta", 0.001))
        self.r = self.p.copy()

        self.oort_reward = np.sqrt(np.maximum(1.0, self.sample_counts))
        self.oort_duration = np.ones(NUM_CLIENTS, dtype=np.float64)
        self.oort_count = np.zeros(NUM_CLIENTS, dtype=np.int64)
        self.oort_last = np.zeros(NUM_CLIENTS, dtype=np.int64)
        self.oort_exploration = float(self.config.get("oort_exploration", 0.2))
        self.oort_exploration_min = 0.05
        self.oort_decay = 0.98
        self.oort_duration_quantile = 0.80
        self.oort_duration_penalty = 2.0

        self.tifl_temperature = float(self.config.get("tifl_temperature", 1.0))
        self.tifl_quality = np.full(5, 0.5, dtype=np.float64)
        credit_share = np.asarray([0.35, 0.25, 0.18, 0.13, 0.09], dtype=np.float64)
        self.tifl_credit = np.maximum(1, np.round(credit_share * round_budget)).astype(np.int64)

    def _uniform_k(self, available: np.ndarray, round_number: int) -> np.ndarray:
        rng = _round_rng(self.seed, round_number, 11)
        return _weighted_choice_without_replacement(rng, available, self.k)

    def _candidate_order(self, available: np.ndarray, round_number: int) -> np.ndarray:
        rng = _round_rng(self.seed, round_number, 23)
        if len(available) <= self.m:
            return np.asarray(available, dtype=np.int64)
        return np.asarray(rng.choice(available, size=self.m, replace=False), dtype=np.int64)

    def admit(
        self,
        available: np.ndarray,
        round_number: int,
        predicted_duration: np.ndarray,
    ) -> Selection:
        available = np.asarray(available, dtype=np.int64)
        scores = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
        admission = np.zeros(NUM_CLIENTS, dtype=np.float64)
        inclusion = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
        tiers = np.full(NUM_CLIENTS, -1, dtype=np.int16)
        if len(available) == 0:
            return Selection(available, available, scores, admission, inclusion, tiers)

        if self.method_id in {"FedAvg", "FedAU", "FedAWE"}:
            selected = self._uniform_k(available, round_number)
            probability = min(1.0, self.k / len(available))
            admission[available] = probability
            inclusion[available] = probability
            scores[available] = self.sample_counts[available]
            return Selection(selected.copy(), selected, scores, admission, inclusion, tiers)

        if self.method_id == "F3AST":
            gradient = np.square(self.p) / np.maximum(np.square(self.r), 1e-18)
            ordered = available[np.lexsort((available, -gradient[available]))]
            selected = ordered[: min(self.k, len(ordered))]
            indicator = np.zeros(NUM_CLIENTS, dtype=np.float64)
            indicator[selected] = 1.0
            self.r = (1.0 - self.f3ast_beta) * self.r + self.f3ast_beta * indicator
            scores[available] = gradient[available]
            admission[selected] = 1.0
            inclusion[selected] = 1.0
            return Selection(selected.copy(), selected, scores, admission, inclusion, tiers)

        if self.method_id == "PowerOfChoice":
            roster = self._candidate_order(available, round_number)
            d = min(int(self.config.get("pow_d", self.m)), len(roster))
            admitted = roster[:d]
            admission[available] = min(1.0, d / len(available))
            return Selection(admitted, np.empty(0, dtype=np.int64), scores, admission, inclusion, tiers)

        if self.method_id == "Oort":
            admitted = self._candidate_order(available, round_number)
            admission[available] = min(1.0, len(admitted) / len(available))
            selected, oort_scores = self._oort_select(admitted, round_number, predicted_duration)
            scores[admitted] = oort_scores
            return Selection(admitted, selected, scores, admission, inclusion, tiers)

        if self.method_id == "TiFL":
            selected, scores, tiers = self._tifl_select(available, round_number, predicted_duration)
            admission[selected] = 1.0
            return Selection(selected.copy(), selected, scores, admission, inclusion, tiers)

        raise ValueError(f"Unsupported baseline {self.method_id}")

    def finish_power_of_choice(self, selection: Selection, probe_losses: dict[int, float]) -> Selection:
        if self.method_id != "PowerOfChoice":
            return selection
        ordered = sorted(selection.admitted.tolist(), key=lambda c: (-float(probe_losses[c]), int(c)))
        selected = np.asarray(ordered[: min(self.k, len(ordered))], dtype=np.int64)
        scores = selection.utility_scores.copy()
        for client_id, value in probe_losses.items():
            scores[int(client_id)] = float(value)
        return Selection(
            admitted=selection.admitted,
            selected=selected,
            utility_scores=scores,
            admission_prob=selection.admission_prob,
            inclusion_prob=selection.inclusion_prob,
            tier_ids=selection.tier_ids,
        )

    def _oort_select(
        self, candidates: np.ndarray, round_number: int, predicted_duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(candidates) <= self.k:
            values = self.oort_reward[candidates].copy()
            return candidates.copy(), values
        durations = np.asarray(predicted_duration, dtype=np.float64)
        self.oort_duration[candidates] = durations[candidates]
        reward = self.oort_reward[candidates]
        lo = float(np.min(reward)) * 0.999
        hi = float(np.quantile(reward, 0.95))
        reward_score = (np.minimum(reward, hi) - lo) / max(1e-8, hi - lo)
        staleness = np.maximum(1, round_number - self.oort_last[candidates])
        temporal = np.sqrt(0.1 * math.log(max(2, round_number + 1)) / np.maximum(1, self.oort_count[candidates]))
        score = reward_score + temporal * np.sqrt(staleness)
        preferred = float(np.quantile(self.oort_duration[candidates], self.oort_duration_quantile))
        penalty = np.ones(len(candidates), dtype=np.float64)
        slow = self.oort_duration[candidates] > preferred
        penalty[slow] = np.power(
            preferred / np.maximum(1e-8, self.oort_duration[candidates][slow]),
            self.oort_duration_penalty,
        )
        score *= penalty
        rng = _round_rng(self.seed, round_number, 31)
        explore_n = min(self.k, max(1, int(round(self.k * self.oort_exploration))))
        unexplored_mask = self.oort_count[candidates] == 0
        unexplored = candidates[unexplored_mask]
        explore = _weighted_choice_without_replacement(
            rng,
            unexplored if len(unexplored) else candidates,
            explore_n,
            self.oort_reward[unexplored] if len(unexplored) else np.maximum(score, 1e-8),
        )
        remaining = np.asarray([c for c in candidates if c not in set(explore.tolist())], dtype=np.int64)
        exploit_n = self.k - len(explore)
        if exploit_n > 0:
            score_map = {int(c): float(score[i]) for i, c in enumerate(candidates)}
            ordered = sorted(remaining.tolist(), key=lambda c: (-score_map[int(c)], int(c)))
            exploit = np.asarray(ordered[:exploit_n], dtype=np.int64)
            selected = np.concatenate([exploit, explore])
        else:
            selected = explore
        self.oort_exploration = max(
            self.oort_exploration_min, self.oort_exploration * self.oort_decay
        )
        return selected, score

    def _tifl_select(
        self, available: np.ndarray, round_number: int, predicted_duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        durations = predicted_duration[available]
        order = available[np.argsort(durations, kind="stable")]
        chunks = np.array_split(order, 5)
        tiers = np.full(NUM_CLIENTS, -1, dtype=np.int16)
        for tier_id, clients in enumerate(chunks):
            tiers[clients] = tier_id
        valid_tiers = [tier for tier, clients in enumerate(chunks) if len(clients) > 0]
        remaining_credit = np.maximum(0, self.tifl_credit[valid_tiers]).astype(np.float64)
        if remaining_credit.sum() == 0:
            remaining_credit[:] = 1.0
        need_weight = np.exp(-self.tifl_quality[valid_tiers] / max(1e-6, self.tifl_temperature))
        probability = remaining_credit * need_weight
        probability /= probability.sum()
        rng = _round_rng(self.seed, round_number, 41)
        chosen_tier = int(rng.choice(np.asarray(valid_tiers), p=probability))
        self.tifl_credit[chosen_tier] = max(0, self.tifl_credit[chosen_tier] - 1)
        chosen = _weighted_choice_without_replacement(rng, chunks[chosen_tier], self.k)
        if len(chosen) < min(self.k, len(available)):
            chosen_set = set(chosen.tolist())
            remainder = np.asarray([c for c in available if c not in chosen_set], dtype=np.int64)
            fill = _weighted_choice_without_replacement(rng, remainder, self.k - len(chosen))
            chosen = np.concatenate([chosen, fill])
        scores = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
        for client_id in available:
            tier = int(tiers[client_id])
            scores[client_id] = -self.tifl_quality[tier] - math.log1p(predicted_duration[client_id])
        self.tier_ids = tiers
        return chosen, scores, tiers

    def aggregation_coefficients(self, eligible: np.ndarray) -> np.ndarray:
        eligible = np.asarray(eligible, dtype=np.int64)
        if len(eligible) == 0:
            return np.empty(0, dtype=np.float64)
        if self.method_id == "FedAU":
            weights = np.asarray(
                [np.mean(self.interval_history[c]) if self.interval_history[c] else 1.0 for c in eligible],
                dtype=np.float64,
            )
            return weights / NUM_CLIENTS
        if self.method_id == "F3AST":
            return self.p[eligible] / np.maximum(self.r[eligible], 1e-12) / self.k
        weights = self.sample_counts[eligible].astype(np.float64)
        return weights / weights.sum()

    def observe(
        self,
        round_number: int,
        selected: np.ndarray,
        eligible: np.ndarray,
        loss_before: dict[int, float],
        loss_after: dict[int, float],
        durations: dict[int, float],
        tier_ids: np.ndarray,
    ) -> None:
        eligible_set = set(int(v) for v in eligible)
        if self.method_id == "FedAU":
            for client_id in range(NUM_CLIENTS):
                if client_id in eligible_set:
                    self.interval_history[client_id].append(int(self.gap[client_id]))
                    self.gap[client_id] = 1
                else:
                    self.gap[client_id] += 1
                    if self.gap[client_id] >= self.fedau_k:
                        self.interval_history[client_id].append(int(self.gap[client_id]))
                        self.gap[client_id] = 1
        if self.method_id == "Oort":
            for client_id in selected:
                client_id = int(client_id)
                if client_id in eligible_set:
                    before = max(1e-8, float(loss_before.get(client_id, 0.0)))
                    after = max(0.0, float(loss_after.get(client_id, before)))
                    # Oort's statistical reward is loss-driven and sample-scaled.
                    self.oort_reward[client_id] = max(
                        1e-8, math.sqrt(self.sample_counts[client_id]) * (0.5 * before + 0.5 * abs(before - after))
                    )
                    self.oort_duration[client_id] = float(durations.get(client_id, 1.0))
                    self.oort_count[client_id] += 1
                    self.oort_last[client_id] = int(round_number)
        if self.method_id == "TiFL":
            tier_values: dict[int, list[float]] = {}
            for client_id in eligible:
                client_id = int(client_id)
                tier = int(tier_ids[client_id])
                if tier < 0:
                    continue
                before = max(1e-8, float(loss_before.get(client_id, 0.0)))
                after = float(loss_after.get(client_id, before))
                improvement = float(np.clip((before - after) / before, -1.0, 1.0))
                tier_values.setdefault(tier, []).append(improvement)
            for tier, values in tier_values.items():
                self.tifl_quality[tier] = 0.9 * self.tifl_quality[tier] + 0.1 * float(np.mean(values))

    def state_dict(self) -> dict[str, Any]:
        return {
            "gap": self.gap,
            "interval_history": self.interval_history,
            "r": self.r,
            "oort_reward": self.oort_reward,
            "oort_duration": self.oort_duration,
            "oort_count": self.oort_count,
            "oort_last": self.oort_last,
            "oort_exploration": self.oort_exploration,
            "tifl_quality": self.tifl_quality,
            "tifl_credit": self.tifl_credit,
        }

