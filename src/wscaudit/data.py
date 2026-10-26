"""Load WSC-Eval-ASR (Kaldi-style ``text``) and write a manifest JSONL.

``wav.scp`` paths point at the authors' machines and are ignored. Local wav
paths are rebuilt as ``WSC-Eval-ASR/<split>/wav/<utt_id>.wav``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import soundfile as sf
from huggingface_hub import snapshot_download

HF_REPO_ID = "ASLP-lab/WSC-Eval"
SPLITS = ("Easy", "Hard")
EXPECTED_COUNTS = {"Easy": 6981, "Hard": 1392}
# Nested globs: a single "*" does not enter wav/.
ALLOW_PATTERNS = ["WSC-Eval-ASR/**"]

_ARABIC_DIGIT = re.compile(r"[0-9]")
_LATIN = re.compile(r"[A-Za-z]")
_HOURS_IN_SUMMARY = re.compile(r"\(([\d.]+)\s*h\)", re.IGNORECASE)


@dataclass(frozen=True)
class RefFlags:
    latin: bool
    space: bool
    arabic_digit: bool
    punctuation: bool


@dataclass
class SplitSummary:
    split: str
    n: int
    hours_sf: float
    hours_txt: float | None
    missing_wavs: int
    non_mono: int
    sample_rates: Counter[int]
    n_latin: int
    n_space: int
    n_digit: int
    n_punct: int
    digit_examples: list[str]
    punct_examples: list[str]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_wsc_eval(data_root: Path, token: str | None = None) -> Path:
    """Snapshot WSC-Eval-ASR into ``data_root``. Resumes if files already exist."""
    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(data_root),
        token=token,
    )
    return data_root


def parse_text_file(path: Path) -> list[tuple[str, str]]:
    """Parse Kaldi ``text``: split on the first space only."""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            if " " not in raw:
                raise ValueError(f"{path}:{line_no}: no space after utt_id: {raw!r}")
            utt_id, ref = raw.split(" ", 1)
            rows.append((utt_id, ref))
    return rows


def parse_duration_txt_hours(path: Path) -> float | None:
    """Read the corpus-level hours figure from the authors' summary file."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    matches = _HOURS_IN_SUMMARY.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def wav_relpath(split: str, utt_id: str) -> str:
    return f"WSC-Eval-ASR/{split}/wav/{utt_id}.wav"


def inspect_ref(ref: str) -> RefFlags:
    return RefFlags(
        latin=bool(_LATIN.search(ref)),
        space=any(ch.isspace() for ch in ref),
        arabic_digit=bool(_ARABIC_DIGIT.search(ref)),
        punctuation=any(unicodedata.category(ch).startswith("P") for ch in ref),
    )


def _audio_info(path: Path) -> tuple[float | None, int | None, int | None]:
    if not path.is_file():
        return None, None, None
    info = sf.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def iter_manifest_rows(
    data_root: Path, splits: Iterable[str] = SPLITS
) -> Iterator[tuple[dict[str, Any], RefFlags, int | None, int | None, bool]]:
    for split in splits:
        text_path = data_root / "WSC-Eval-ASR" / split / "text"
        if not text_path.is_file():
            raise FileNotFoundError(f"missing {text_path}")
        for utt_id, ref in parse_text_file(text_path):
            rel = wav_relpath(split, utt_id)
            wav_path = data_root / rel
            dur, sr, channels = _audio_info(wav_path)
            missing = dur is None
            row = {
                "utt_id": utt_id,
                "split": split,
                "wav": rel,
                "ref": ref,
                "dur": dur,
            }
            yield row, inspect_ref(ref), sr, channels, missing


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _empty_acc(split: str, data_root: Path) -> dict[str, Any]:
    return {
        "split": split,
        "n": 0,
        "dur_sum": 0.0,
        "hours_txt": parse_duration_txt_hours(
            data_root / "WSC-Eval-ASR" / split / "duration.txt"
        ),
        "missing_wavs": 0,
        "non_mono": 0,
        "sample_rates": Counter(),
        "n_latin": 0,
        "n_space": 0,
        "n_digit": 0,
        "n_punct": 0,
        "digit_examples": [],
        "punct_examples": [],
    }


def _acc_to_summary(acc: dict[str, Any]) -> SplitSummary:
    return SplitSummary(
        split=acc["split"],
        n=acc["n"],
        hours_sf=acc["dur_sum"] / 3600.0,
        hours_txt=acc["hours_txt"],
        missing_wavs=acc["missing_wavs"],
        non_mono=acc["non_mono"],
        sample_rates=acc["sample_rates"],
        n_latin=acc["n_latin"],
        n_space=acc["n_space"],
        n_digit=acc["n_digit"],
        n_punct=acc["n_punct"],
        digit_examples=acc["digit_examples"],
        punct_examples=acc["punct_examples"],
    )


def format_summary(summaries: list[SplitSummary], manifest_path: Path, n_written: int) -> str:
    lines = ["=== WSC-Eval-ASR prepare ==="]
    rates_all: Counter[int] = Counter()
    for s in summaries:
        expected = EXPECTED_COUNTS.get(s.split)
        count_note = ""
        if expected is not None and s.n != expected:
            count_note = f"  ** count != expected {expected} **"
        txt = "n/a" if s.hours_txt is None else f"{s.hours_txt:.2f}"
        lines.append(
            f"{s.split}: n={s.n}{count_note}  hours_sf={s.hours_sf:.3f}  "
            f"hours_duration_txt={txt}  missing_wavs={s.missing_wavs}  non_mono={s.non_mono}"
        )
        lines.append(
            f"  refs: latin={s.n_latin}  spaces={s.n_space}  "
            f"arabic_digits={s.n_digit}  punctuation={s.n_punct}"
        )
        if s.n_digit or s.n_punct:
            lines.append(
                "  note: earlier inspection expected 0 Arabic digits and 0 punctuation; "
                "reporting the difference rather than altering refs."
            )
            if s.digit_examples:
                lines.append(f"  arabic_digit examples: {s.digit_examples}")
            if s.punct_examples:
                lines.append(f"  punctuation examples: {s.punct_examples}")
        rates_all.update(s.sample_rates)
    lines.append(f"sample_rates_seen: {dict(rates_all)}")
    lines.append(f"wrote {n_written} lines -> {manifest_path}")
    return "\n".join(lines)


def prepare(
    data_root: Path,
    out_root: Path,
    *,
    skip_download: bool = False,
    token: str | None = None,
    splits: Iterable[str] = SPLITS,
) -> Path:
    """Download (unless skipped), write ``out_root/manifest.jsonl``, print a summary."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    if not skip_download:
        download_wsc_eval(data_root, token=token if token is not None else hf_token())

    split_list = tuple(splits)
    acc = {s: _empty_acc(s, data_root) for s in split_list}
    ordered: list[dict[str, Any]] = []
    for row, flags, sr, channels, missing in iter_manifest_rows(data_root, split_list):
        a = acc[row["split"]]
        a["n"] += 1
        a["n_latin"] += int(flags.latin)
        a["n_space"] += int(flags.space)
        a["n_digit"] += int(flags.arabic_digit)
        a["n_punct"] += int(flags.punctuation)
        if flags.arabic_digit and len(a["digit_examples"]) < 5:
            a["digit_examples"].append(row["utt_id"])
        if flags.punctuation and len(a["punct_examples"]) < 5:
            a["punct_examples"].append(row["utt_id"])
        if missing:
            a["missing_wavs"] += 1
        else:
            a["dur_sum"] += float(row["dur"])
            if sr is not None:
                a["sample_rates"][sr] += 1
            if channels is not None and channels != 1:
                a["non_mono"] += 1
        ordered.append(row)

    manifest_path = out_root / "manifest.jsonl"
    n_written = write_jsonl(manifest_path, ordered)
    summaries = [_acc_to_summary(acc[s]) for s in split_list]
    print(format_summary(summaries, manifest_path, n_written))
    return manifest_path
