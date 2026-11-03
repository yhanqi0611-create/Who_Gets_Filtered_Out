"""Holm adjustment."""

from __future__ import annotations

from statsmodels.stats.multitest import multipletests


def holm_adjust(pvals: list[float]) -> list[float]:
    arr = [1.0 if (p is None or p != p) else float(min(1.0, max(0.0, p))) for p in pvals]
    if not arr:
        return []
    _, adj, _, _ = multipletests(arr, method="holm")
    return [float(x) for x in adj]
