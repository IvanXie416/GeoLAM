"""Sampling queries from canonical query pools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuerySampler:
    queries_per_clip: int
    hard_query_ratio: float = 0.0
    prob_t_tgt_equals_t_cam: float = 0.0
    training: bool = True

    def _random_choice(self, ids: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        if ids.size == 0:
            return np.empty((0,), dtype=np.int64)
        replace = ids.size < n
        return rng.choice(ids, size=n, replace=replace)

    def sample_indices(self, query_pool: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        total = query_pool["q_u_src"].shape[0]
        if total <= 0:
            raise ValueError("query_pool is empty")

        if not self.training:
            if total >= self.queries_per_clip:
                return np.arange(self.queries_per_clip, dtype=np.int64)
            extra = rng.choice(np.arange(total), size=self.queries_per_clip - total, replace=True)
            return np.concatenate([np.arange(total), extra.astype(np.int64)], axis=0)

        all_ids = np.arange(total, dtype=np.int64)
        n_hard = int(round(self.queries_per_clip * self.hard_query_ratio))
        n_easy = self.queries_per_clip - n_hard

        hard_mask = query_pool.get("is_hard_boundary_query")
        hard_ids = all_ids[hard_mask.astype(bool)] if hard_mask is not None else np.empty((0,), dtype=np.int64)
        easy_ids = all_ids[~hard_mask.astype(bool)] if hard_mask is not None else all_ids

        chosen_hard = self._random_choice(hard_ids, n_hard, rng) if n_hard > 0 else np.empty((0,), dtype=np.int64)
        chosen_easy = self._random_choice(easy_ids, n_easy, rng) if n_easy > 0 else np.empty((0,), dtype=np.int64)
        sampled = np.concatenate([chosen_hard, chosen_easy], axis=0)

        if sampled.size < self.queries_per_clip:
            fill = self._random_choice(all_ids, self.queries_per_clip - sampled.size, rng)
            sampled = np.concatenate([sampled, fill], axis=0)

        if self.prob_t_tgt_equals_t_cam > 0.0:
            target_eq = int(round(self.queries_per_clip * self.prob_t_tgt_equals_t_cam))
            target_eq = int(np.clip(target_eq, 0, self.queries_per_clip))

            eq_mask_all = query_pool["q_t_tgt"] == query_pool["q_t_cam"]
            eq_ids = all_ids[eq_mask_all]
            neq_ids = all_ids[~eq_mask_all]
            if eq_ids.size > 0 and neq_ids.size > 0:
                sampled_eq = eq_mask_all[sampled]
                current_eq = int(sampled_eq.sum())
                if current_eq < target_eq:
                    need = target_eq - current_eq
                    replace_candidates = np.flatnonzero(~sampled_eq)
                    if replace_candidates.size > 0:
                        replace_idx = rng.choice(replace_candidates, size=min(need, replace_candidates.size), replace=False)
                        sampled[replace_idx] = self._random_choice(eq_ids, replace_idx.size, rng)
                elif current_eq > target_eq:
                    need = current_eq - target_eq
                    replace_candidates = np.flatnonzero(sampled_eq)
                    if replace_candidates.size > 0:
                        replace_idx = rng.choice(replace_candidates, size=min(need, replace_candidates.size), replace=False)
                        sampled[replace_idx] = self._random_choice(neq_ids, replace_idx.size, rng)

        rng.shuffle(sampled)
        return sampled.astype(np.int64)
