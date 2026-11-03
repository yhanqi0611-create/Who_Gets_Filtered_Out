"""Within-split permutation of has_dialect_ref for pass-rate differences."""

from __future__ import annotations

import numpy as np


def pass_delta_pp(has: np.ndarray, passed: np.ndarray) -> float:
    has_b = has.astype(bool)
    n1 = int(has_b.sum())
    n0 = int((~has_b).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r1 = float(passed[has_b].mean())
    r0 = float(passed[~has_b].mean())
    return (r1 - r0) * 100.0


def permute_has_within_split(
    has: np.ndarray,
    splits: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(B, n) shuffled copies of ``has``, permuted independently inside each split."""
    n = has.size
    out = np.empty((B, n), dtype=bool)
    out[:, :] = has
    for spl in ("Easy", "Hard"):
        idx = np.flatnonzero(splits == spl)
        k = int(idx.size)
        if k == 0:
            continue
        order = np.argsort(rng.random((B, k)), axis=1)
        out[:, idx] = has[idx][order]
    return out


def permutation_p(obs: float, nulls: np.ndarray, B: int) -> float:
    if not np.isfinite(obs):
        return float("nan")
    p = float(np.mean(np.abs(nulls) >= abs(obs)))
    return float(min(1.0, max(1.0 / B, p)))
