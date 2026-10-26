from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wscaudit.data import (
    inspect_ref,
    parse_duration_txt_hours,
    parse_text_file,
    prepare,
    wav_relpath,
)


def _write_wav(path: Path, seconds: float = 0.25, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    sf.write(str(path), 0.01 * np.sin(2 * np.pi * 440 * t), sr)


def _layout(root: Path) -> Path:
    easy = root / "WSC-Eval-ASR" / "Easy"
    hard = root / "WSC-Eval-ASR" / "Hard"
    (easy / "wav").mkdir(parents=True)
    (hard / "wav").mkdir(parents=True)
    (easy / "text").write_text(
        "utt_e1 诶呀好巴适\n"
        "utt_e2 hello 世界\n",
        encoding="utf-8",
    )
    (hard / "text").write_text("utt_h1 瓜娃子\n", encoding="utf-8")
    (easy / "duration.txt").write_text(
        "====== Summary ======\nTotal duration: 0h 0min (0.00h)\n",
        encoding="utf-8",
    )
    (hard / "duration.txt").write_text(
        "====== Summary ======\nTotal duration: 0h 0min (0.00h)\n",
        encoding="utf-8",
    )
    _write_wav(easy / "wav" / "utt_e1.wav", seconds=0.5)
    _write_wav(easy / "wav" / "utt_e2.wav", seconds=0.25)
    _write_wav(hard / "wav" / "utt_h1.wav", seconds=1.0)
    return root


def test_parse_text_splits_on_first_space_only(tmp_path: Path) -> None:
    p = tmp_path / "text"
    p.write_text("u1 foo bar baz\n", encoding="utf-8")
    assert parse_text_file(p) == [("u1", "foo bar baz")]


def test_inspect_ref_flags() -> None:
    latin = inspect_ref("hello世界")
    assert latin.latin and not latin.arabic_digit and not latin.punctuation
    spaced = inspect_ref("好 巴适")
    assert spaced.space
    digits = inspect_ref("3个")
    assert digits.arabic_digit
    punct = inspect_ref("好巴适。")
    assert punct.punctuation


def test_prepare_skip_download_writes_manifest(tmp_path: Path) -> None:
    data = _layout(tmp_path / "data")
    out = tmp_path / "out"
    path = prepare(data, out, skip_download=True)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    assert rows[0]["utt_id"] == "utt_e1"
    assert rows[0]["wav"] == wav_relpath("Easy", "utt_e1")
    assert rows[0]["ref"] == "诶呀好巴适"
    assert rows[0]["dur"] == pytest.approx(0.5, abs=0.02)
    assert rows[1]["ref"] == "hello 世界"
    assert {r["split"] for r in rows} == {"Easy", "Hard"}


def test_parse_duration_txt_hours(tmp_path: Path) -> None:
    p = tmp_path / "duration.txt"
    p.write_text(
        "====== Summary ======\n"
        "files: 6981\n"
        "Total duration: 8h 32min 51.02s (8.55h)\n",
        encoding="utf-8",
    )
    assert parse_duration_txt_hours(p) == pytest.approx(8.55)
