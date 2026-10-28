"""FireRedASR-AED-L wrapper (third_party/FireRedASR)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIRERED_SRC = _REPO_ROOT / "third_party" / "FireRedASR"

# README example decode knobs (Prompt 4).
DEFAULT_DECODE = {
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.25,
    "aed_length_penalty": 0.6,
    "eos_penalty": 1.0,
}


def ensure_firered_on_path() -> Path:
    if not _FIRERED_SRC.is_dir():
        raise FileNotFoundError(
            "FireRedASR checkout missing. From the repo root run:\n"
            "  git clone https://github.com/FireRedTeam/FireRedASR.git "
            "third_party/FireRedASR\n"
            "  pip install -r third_party/FireRedASR/requirements.txt"
        )
    src = str(_FIRERED_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return _FIRERED_SRC


def _texts_in_order(utt_ids: list[str], results: list[dict[str, Any]]) -> list[str]:
    by_id: dict[str, str] = {}
    ordered: list[str] = []
    for item in results:
        uid = str(item.get("uttid") or item.get("utt_id") or "")
        text = item.get("text")
        if uid:
            by_id[uid] = "" if text is None else str(text)
        else:
            ordered.append("" if text is None else str(text))
    if len(by_id) == len(utt_ids):
        return [by_id.get(u, "") for u in utt_ids]
    if len(ordered) == len(utt_ids):
        return ordered
    if len(results) == len(utt_ids):
        return ["" if r.get("text") is None else str(r.get("text")) for r in results]
    raise RuntimeError(
        f"FireRed returned {len(results)} results for {len(utt_ids)} wavs"
    )


class FireRedEngine:
    """AED-L only. Weights from HF; code from third_party/FireRedASR."""

    def __init__(self, device: str, cfg: dict[str, Any] | None = None) -> None:
        ensure_firered_on_path()
        from fireredasr.models.fireredasr import FireRedAsr

        cfg = cfg or {}
        self.hf_id = cfg.get("hf_id") or "FireRedTeam/FireRedASR-AED-L"
        self.asr_type = cfg.get("asr_type") or "aed"
        self.use_gpu = 1 if str(device).startswith("cuda") else 0
        self.decode = {**DEFAULT_DECODE}
        for key in DEFAULT_DECODE:
            if key in cfg:
                self.decode[key] = cfg[key]
        self.decode["use_gpu"] = self.use_gpu
        model_dir = cfg.get("model_dir")
        if model_dir:
            self.model_dir = str(model_dir)
        else:
            self.model_dir = snapshot_download(self.hf_id)
        self.model = FireRedAsr.from_pretrained(self.asr_type, self.model_dir)

    def transcribe_batch(
        self, wavs: list[str], utt_ids: list[str] | None = None
    ) -> list[str]:
        if not wavs:
            return []
        ids = utt_ids or [Path(w).stem for w in wavs]
        if len(ids) != len(wavs):
            raise ValueError("utt_ids and wavs length mismatch")
        results = self.model.transcribe(ids, wavs, dict(self.decode))
        if results is None:
            return [""] * len(wavs)
        return _texts_in_order(ids, list(results))
