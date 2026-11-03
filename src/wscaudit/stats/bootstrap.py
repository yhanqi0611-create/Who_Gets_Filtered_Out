"""Utterance-level bootstrap (stratified by split for All)."""

from __future__ import annotations

import numpy as np

SEED = 20261103


def stratified_indices(
    splits: np.ndarray,
    B: int,
    rng: np.random.Generator,
    *,
    split_filter: str | None = None,
) -> np.ndarray:
    """Return (B, n_out) integer indices into the original arrays.

    If ``split_filter`` is Easy/Hard, resample only that split.
    If None, resample Easy and Hard separately and concatenate (All).
    """
    if split_filter in {"Easy", "Hard"}:
        idx = np.flatnonzero(splits == split_filter)
        n = int(idx.size)
        draw = rng.integers(0, n, size=(B, n))
        return idx[draw]
    easy = np.flatnonzero(splits == "Easy")
    hard = np.flatnonzero(splits == "Hard")
    e_n, h_n = int(easy.size), int(hard.size)
    e_draw = easy[rng.integers(0, e_n, size=(B, e_n))] if e_n else np.zeros((B, 0), dtype=int)
    h_draw = hard[rng.integers(0, h_n, size=(B, h_n))] if h_n else np.zeros((B, 0), dtype=int)
    return np.concatenate([e_draw, h_draw], axis=1)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(samples.astype(float), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_p(samples: np.ndarray, B: int) -> float:
    """Two-sided: 2 * min(P(s>=0), P(s<=0)), floored at 1/B, capped at 1."""
    s = np.asarray(samples, dtype=float)
    ge = float(np.mean(s >= 0.0))
    le = float(np.mean(s <= 0.0))
    p = 2.0 * min(ge, le)
    return float(min(1.0, max(1.0 / B, p)))


def micro_cer_batch(distance: np.ndarray, ref_len: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = distance[idx].sum(axis=1)
    n = ref_len[idx].sum(axis=1).astype(float)
    out = np.full(idx.shape[0], np.nan)
    ok = n > 0
    out[ok] = d[ok] / n[ok]
    return out
