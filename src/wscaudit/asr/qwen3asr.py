"""Qwen3-ASR-1.7B wrapper (TeleASR substitute)."""

from __future__ import annotations


def transcribe(*_args, **_kwargs) -> None:
    raise NotImplementedError("qwen3asr: implemented in Prompt 4")
