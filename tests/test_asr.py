from __future__ import annotations

import json
from pathlib import Path

from wscaudit.asr.base import run_asr
from wscaudit.asr.firered import _texts_in_order
from wscaudit.data import prepare
from wscaudit.normalize import normalize


def _tagged(wavs: list[str], utt_ids: list[str] | None = None) -> list[str]:
    return ["<|zh|><|NEUTRAL|>巴适"] * len(wavs)


def _tiny_manifest(tmp_path: Path) -> tuple[Path, Path]:
    import numpy as np
    import soundfile as sf

    data = tmp_path / "data"
    easy = data / "WSC-Eval-ASR" / "Easy"
    hard = data / "WSC-Eval-ASR" / "Hard"
    (easy / "wav").mkdir(parents=True)
    (hard / "wav").mkdir(parents=True)
    (easy / "text").write_text("utt_e1 诶呀好巴适\nutt_e2 hello 世界\n", encoding="utf-8")
    (hard / "text").write_text("utt_h1 瓜娃子\n", encoding="utf-8")
    (easy / "duration.txt").write_text("Total duration: (0.00h)\n", encoding="utf-8")
    (hard / "duration.txt").write_text("Total duration: (0.00h)\n", encoding="utf-8")
    for split, utt, sec in (("Easy", "utt_e1", 0.5), ("Easy", "utt_e2", 0.25), ("Hard", "utt_h1", 1.0)):
        n = int(16000 * sec)
        t = np.arange(n, dtype=np.float32) / 16000
        sf.write(str(data / "WSC-Eval-ASR" / split / "wav" / f"{utt}.wav"), 0.01 * np.sin(2 * np.pi * 440 * t), 16000)
    out = tmp_path / "out"
    manifest = prepare(data, out, skip_download=True)
    return data, manifest


def test_run_asr_limit_and_norm(tmp_path: Path) -> None:
    data, manifest = _tiny_manifest(tmp_path)
    out = tmp_path / "out"
    hyps = out / "hyps" / "sensevoice.jsonl"
    run_asr(
        "sensevoice",
        manifest,
        hyps,
        data,
        batch_size=2,
        device="cpu",
        split="all",
        limit=2,
        transcribe_batch=_tagged,
        runtime_csv=out / "runtime.csv",
    )
    rows = [json.loads(x) for x in hyps.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(r["raw"].startswith("<|zh|>") for r in rows)
    assert all(r["norm"] == normalize("<|zh|><|NEUTRAL|>巴适") for r in rows)
    assert (out / "runtime.csv").is_file()


def test_run_asr_idempotent_skips(tmp_path: Path) -> None:
    data, manifest = _tiny_manifest(tmp_path)
    hyps = tmp_path / "out" / "hyps" / "sensevoice.jsonl"
    kwargs = dict(
        model_name="sensevoice",
        manifest=manifest,
        out_path=hyps,
        data_root=data,
        batch_size=8,
        device="cpu",
        transcribe_batch=_tagged,
    )
    run_asr(**kwargs)
    run_asr(**kwargs)
    lines = hyps.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_firered_aligns_results_by_uttid() -> None:
    texts = _texts_in_order(
        ["u2", "u1"],
        [{"uttid": "u1", "text": "一"}, {"uttid": "u2", "text": "二"}],
    )
    assert texts == ["二", "一"]
