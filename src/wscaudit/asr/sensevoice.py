"""SenseVoice-Small wrapper (funasr)."""

from __future__ import annotations

from typing import Any


def _item_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "pred", "value"):
            if key in item and item[key] is not None:
                return str(item[key])
        return str(item)
    return str(item)


def postprocess_sensevoice(text: str) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return rich_transcription_postprocess(text)
    except Exception:
        return text


class SenseVoiceEngine:
    """Lazy FunASR loader. Short clips: no VAD (model-card recommendation)."""

    def __init__(self, device: str, cfg: dict[str, Any] | None = None) -> None:
        from funasr import AutoModel

        cfg = cfg or {}
        self.hf_id = cfg.get("hf_id") or "FunAudioLLM/SenseVoiceSmall"
        self.language = cfg.get("language") or "zh"
        self.use_itn = bool(cfg.get("use_itn", False))
        self.batch_size = int(cfg.get("batch_size") or 64)
        self.model = AutoModel(
            model=self.hf_id,
            device=device,
            hub="hf",
        )

    def _generate(self, inp: str | list[str], n: int) -> list[str]:
        kwargs = {
            "input": inp,
            "cache": {},
            "language": self.language,
            "use_itn": self.use_itn,
            "batch_size": n,
        }
        try:
            res = self.model.generate(**kwargs)
        except TypeError:
            kwargs.pop("batch_size", None)
            res = self.model.generate(**kwargs)
        if res is None:
            return [""] if n == 1 else [""] * (n if isinstance(inp, list) else 1)
        if isinstance(res, dict):
            items = [res]
        else:
            items = list(res)
        return [postprocess_sensevoice(_item_text(it)) for it in items]

    def transcribe_batch(self, wavs: list[str]) -> list[str]:
        if not wavs:
            return []
        texts = self._generate(wavs, min(self.batch_size, len(wavs)))
        if len(texts) == len(wavs):
            return texts
        # Some FunASR builds ignore list input / batching; fall back per file.
        out: list[str] = []
        for wav in wavs:
            one = self._generate(wav, 1)
            out.append(one[0] if one else "")
        return out
