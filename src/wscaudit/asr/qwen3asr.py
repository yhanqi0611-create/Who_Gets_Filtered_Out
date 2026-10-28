"""Qwen3-ASR-1.7B wrapper (TeleASR substitute)."""

from __future__ import annotations

from typing import Any

from wscaudit.asr.base import resolve_device


def _torch_dtype(name: str):
    import torch

    key = (name or "bfloat16").lower()
    if key in {"bfloat16", "bf16"}:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"float32", "fp32"}:
        return torch.float32
    return getattr(torch, name)


def _as_result_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    return [results]


def _item_text_lang(item: Any) -> tuple[str, str | None]:
    if item is None:
        return "", None
    if isinstance(item, str):
        return item, None
    text = getattr(item, "text", None)
    lang = getattr(item, "language", None)
    if text is None and isinstance(item, dict):
        text = item.get("text")
        lang = item.get("language")
    return ("" if text is None else str(text), None if lang is None else str(lang))


class Qwen3ASREngine:
    """Transformers backend by default; ``backend=vllm`` uses ``Qwen3ASRModel.LLM``."""

    def __init__(self, device: str, cfg: dict[str, Any] | None = None) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        cfg = cfg or {}
        self.hf_id = cfg.get("hf_id") or "Qwen/Qwen3-ASR-1.7B"
        self.backend = (cfg.get("backend") or "transformers").lower()
        self.language = cfg.get("language")  # None = auto (card lists Sichuan)
        max_batch = int(cfg.get("max_inference_batch_size") or 32)
        max_new = int(cfg.get("max_new_tokens") or 256)
        device = resolve_device(device)
        dtype = _torch_dtype(str(cfg.get("dtype") or "bfloat16"))

        if self.backend in {"vllm", "llm"}:
            self.model = Qwen3ASRModel.LLM(
                model=self.hf_id,
                max_inference_batch_size=max_batch,
                max_new_tokens=max_new,
            )
        else:
            kwargs: dict[str, Any] = {
                "dtype": dtype,
                "max_inference_batch_size": max_batch,
                "max_new_tokens": max_new,
            }
            if device != "cpu":
                kwargs["device_map"] = device
            else:
                kwargs["device_map"] = "cpu"
                kwargs["dtype"] = torch.float32
            self.model = Qwen3ASRModel.from_pretrained(self.hf_id, **kwargs)

    def transcribe_batch(
        self, wavs: list[str], utt_ids: list[str] | None = None
    ) -> list[dict[str, str]]:
        if not wavs:
            return []
        results = self.model.transcribe(audio=wavs, language=self.language)
        items = _as_result_list(results)
        if len(items) != len(wavs):
            raise RuntimeError(
                f"Qwen3-ASR returned {len(items)} hyps for {len(wavs)} wavs"
            )
        out: list[dict[str, str]] = []
        for item in items:
            text, lang = _item_text_lang(item)
            rec: dict[str, str] = {"text": text}
            if lang:
                rec["language"] = lang
            out.append(rec)
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-load Qwen3-ASR (optional vLLM).")
    parser.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    engine = Qwen3ASREngine(
        args.device,
        {"hf_id": args.model, "backend": args.backend},
    )
    print("loaded", args.backend, type(engine.model).__name__)
