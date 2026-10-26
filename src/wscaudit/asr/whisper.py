"""Whisper-large-v3 wrapper (sensitivity check S1)."""

from __future__ import annotations


def transcribe(*_args, **_kwargs) -> None:
    raise NotImplementedError("whisper: implemented in Prompt 4")
