"""Shared text normalisation for refs and hypotheses."""

from __future__ import annotations

from typing import Any


def normalize(text: str, cfg: dict[str, Any] | None = None) -> str:
    raise NotImplementedError("normalize: implemented in Prompt 3")
