"""Shared ASR runner: duration-sorted batches, idempotent JSONL writes."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from wscaudit.normalize import load_normalization_cfg, normalize

TranscribeBatch = Callable[..., list[str]]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_utt_ids(path: Path) -> set[str]:
    return {row["utt_id"] for row in load_jsonl(path) if "utt_id" in row}


def resolve_wav(data_root: Path, wav: str) -> Path:
    p = Path(wav)
    return p if p.is_absolute() else (data_root / wav)


def corpus_cer(pairs: Iterable[tuple[str, str]]) -> float | None:
    edits = 0
    n_ref = 0
    for ref, hyp in pairs:
        n_ref += len(ref)
        edits += Levenshtein.distance(ref, hyp)
    if n_ref == 0:
        return None
    return edits / n_ref


def _filter_manifest(
    rows: list[dict[str, Any]],
    split: str,
    limit: int | None,
    skip: set[str],
) -> list[dict[str, Any]]:
    split_l = split.lower()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["utt_id"] in skip:
            continue
        if split_l not in {"all", ""} and str(row.get("split", "")).lower() != split_l:
            continue
        out.append(row)
    out.sort(key=lambda r: (float(r["dur"]) if r.get("dur") is not None else 0.0, r["utt_id"]))
    if limit is not None:
        out = out[:limit]
    return out


def _upsert_runtime_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n_written",
        "n_skipped",
        "n_empty_raw",
        "runtime_s",
        "cer_hard",
        "split",
        "limit",
        "device",
        "timestamp",
    ]
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("model") != record["model"]]
    rows.append({k: "" if record.get(k) is None else str(record[k]) for k in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_transcriber(model_name: str, device: str, asr_cfg: dict[str, Any]) -> TranscribeBatch:
    if model_name == "sensevoice":
        from wscaudit.asr.sensevoice import SenseVoiceEngine

        engine = SenseVoiceEngine(device=device, cfg=asr_cfg.get("sensevoice") or {})
        return engine.transcribe_batch
    if model_name == "firered":
        from wscaudit.asr.firered import FireRedEngine

        engine = FireRedEngine(device=device, cfg=asr_cfg.get("firered") or {})
        return engine.transcribe_batch
    raise NotImplementedError(
        f"asr model {model_name!r} is not implemented yet (sensevoice|firered)"
    )


def run_asr(
    model_name: str,
    manifest: Path,
    out_path: Path,
    data_root: Path,
    *,
    batch_size: int = 64,
    device: str = "auto",
    split: str = "all",
    limit: int | None = None,
    norm_cfg: dict[str, Any] | None = None,
    transcribe_batch: TranscribeBatch | None = None,
    asr_cfg: dict[str, Any] | None = None,
    runtime_csv: Path | None = None,
) -> Path:
    """Transcribe remaining manifest rows; skip ``utt_id``s already in ``out_path``."""
    device = resolve_device(device)
    norm_cfg = norm_cfg if norm_cfg is not None else load_normalization_cfg()
    rows = load_jsonl(manifest)
    if not rows:
        raise FileNotFoundError(f"empty or missing manifest: {manifest}")

    skip = existing_utt_ids(out_path)
    todo = _filter_manifest(rows, split, limit, skip)
    n_skipped = len(skip)
    print(
        f"asr {model_name}: device={device} split={split} "
        f"todo={len(todo)} skipped={n_skipped} batch_size={batch_size}"
    )

    if transcribe_batch is None:
        transcribe_batch = load_transcriber(model_name, device, asr_cfg or {})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_empty = 0
    t0 = time.perf_counter()
    by_id = {r["utt_id"]: r for r in rows}

    with out_path.open("a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(todo), batch_size), desc=f"asr:{model_name}", leave=True):
            batch = todo[i : i + batch_size]
            wavs: list[str] = []
            utt_ids: list[str] = []
            for row in batch:
                wav_path = resolve_wav(data_root, row["wav"])
                wavs.append(str(wav_path))
                utt_ids.append(row["utt_id"])
            b0 = time.perf_counter()
            hyps = transcribe_batch(wavs, utt_ids)
            batch_s = time.perf_counter() - b0
            if len(hyps) != len(batch):
                raise RuntimeError(
                    f"{model_name} returned {len(hyps)} hyps for {len(batch)} wavs"
                )
            per_utt = batch_s / max(len(batch), 1)
            for row, raw in zip(batch, hyps, strict=True):
                raw = "" if raw is None else str(raw)
                if not raw.strip():
                    n_empty += 1
                    print(f"warning: empty raw for {row['utt_id']}", flush=True)
                rec = {
                    "utt_id": row["utt_id"],
                    "model": model_name,
                    "raw": raw,
                    "norm": normalize(raw, norm_cfg),
                    "runtime_s": per_utt,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            fout.flush()

    elapsed = time.perf_counter() - t0
    all_hyps = load_jsonl(out_path)
    hard_pairs: list[tuple[str, str]] = []
    for hyp in all_hyps:
        src = by_id.get(hyp["utt_id"])
        if src is None or str(src.get("split")) != "Hard":
            continue
        hard_pairs.append((normalize(src["ref"], norm_cfg), hyp.get("norm") or ""))
    cer_hard = corpus_cer(hard_pairs)
    cer_txt = "n/a" if cer_hard is None else f"{cer_hard:.4f}"
    print(
        f"asr {model_name}: wrote={n_written} empty_raw={n_empty} "
        f"runtime_s={elapsed:.1f} cer_hard={cer_txt} (n={len(hard_pairs)}) -> {out_path}"
    )
    if runtime_csv is not None:
        _upsert_runtime_csv(
            runtime_csv,
            {
                "model": model_name,
                "n_written": n_written,
                "n_skipped": n_skipped,
                "n_empty_raw": n_empty,
                "runtime_s": round(elapsed, 3),
                "cer_hard": None if cer_hard is None else round(cer_hard, 6),
                "split": split,
                "limit": limit,
                "device": device,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    return out_path
