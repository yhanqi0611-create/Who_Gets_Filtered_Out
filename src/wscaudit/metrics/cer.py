"""Character error rate tables (Prompt 8). CPU only.

Per-utterance: Levenshtein(hyp, ref) / max(1, len(ref)) on normalised strings
(rapidfuzz, equal sub/ins/del). Corpus metric is **micro** CER:
sum(distance) / sum(ref_len). Equivalences are not applied.

Empty/missing recogniser ``norm`` (or empty vote ``label``) is a full deletion:
distance = len(ref); CER = 1.0 if the ref is non-empty, else 0.0. The utterance
stays in n.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz.distance import Levenshtein

from wscaudit.asr.base import load_jsonl
from wscaudit.normalize import load_normalization_cfg, normalize

DEFAULT_SYSTEMS = ("sensevoice", "firered", "qwen3asr", "vote_P0")
DISPLAY = {
    "sensevoice": "SenseVoice",
    "firered": "FireRed",
    "qwen3asr": "Qwen3-ASR",
    "vote_P0": "vote_P0",
}
SPLITS = ("Easy", "Hard", "All")
CONF_BINS = (
    ("<0.6", 0.0, 0.6),
    ("0.6-0.9", 0.6, 0.9),
    (">=0.9", 0.9, 1.01),
)


def hyp_is_empty(text: str | None) -> bool:
    return text is None or not str(text).strip()


def utt_distance_cer(hyp: str | None, ref: str) -> tuple[int, float, bool]:
    """Return (distance, per-utt CER, empty_hyp)."""
    ref_s = ref or ""
    empty = hyp_is_empty(hyp)
    if empty:
        distance = len(ref_s)
        cer = 0.0 if not ref_s else 1.0
        return distance, cer, True
    distance = int(Levenshtein.distance(str(hyp), ref_s))
    return distance, distance / max(1, len(ref_s)), False


def micro_cer(distance_sum: int, ref_len_sum: int) -> float | None:
    if ref_len_sum <= 0:
        return None
    return distance_sum / ref_len_sum


def conf_bin(conf: float | None) -> str | None:
    if conf is None:
        return None
    c = float(conf)
    if c < 0.6:
        return "<0.6"
    if c < 0.9:
        return "0.6-0.9"
    return ">=0.9"


def _index_by_utt(path: Path) -> dict[str, dict[str, Any]]:
    return {row["utt_id"]: row for row in load_jsonl(path) if "utt_id" in row}


def _system_hyp(
    system: str,
    rec: dict[str, Any] | None,
) -> tuple[str | None, float | None]:
    if rec is None:
        return None, None
    if system.startswith("vote"):
        label = rec.get("label")
        conf = rec.get("conf")
        text = None if hyp_is_empty(None if label is None else str(label)) else str(label)
        conf_f = None if conf is None else float(conf)
        return text, conf_f
    norm = rec.get("norm")
    if hyp_is_empty(None if norm is None else str(norm)):
        return None, None
    return str(norm), None


def _system_path(out_root: Path, system: str) -> Path:
    if system.startswith("vote"):
        return out_root / "merged" / f"{system}.jsonl"
    return out_root / "hyps" / f"{system}.jsonl"


def _blank_cell() -> dict[str, Any]:
    return {"n": 0, "n_empty_hyp": 0, "distance_sum": 0, "ref_len_sum": 0, "cer": None}


def _add(cell: dict[str, Any], distance: int, ref_len: int, empty: bool) -> None:
    cell["n"] += 1
    cell["n_empty_hyp"] += int(empty)
    cell["distance_sum"] += distance
    cell["ref_len_sum"] += ref_len
    cell["cer"] = micro_cer(cell["distance_sum"], cell["ref_len_sum"])


def _finalise(cell: dict[str, Any]) -> dict[str, Any]:
    out = dict(cell)
    cer = micro_cer(out["distance_sum"], out["ref_len_sum"])
    out["cer"] = None if cer is None else round(cer, 6)
    return out


def run_cer(
    manifest_path: Path,
    out_root: Path,
    systems: Iterable[str] = DEFAULT_SYSTEMS,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    systems = tuple(systems) or DEFAULT_SYSTEMS
    metrics_dir = metrics_dir or (out_root / "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_jsonl(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"empty or missing manifest: {manifest_path}")

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    missing_files: list[Path] = []
    for system in systems:
        path = _system_path(out_root, system)
        if not path.is_file():
            missing_files.append(path)
            continue
        indexed[system] = _index_by_utt(path)
    if missing_files:
        raise FileNotFoundError("missing system files:\n  " + "\n  ".join(str(p) for p in missing_files))

    norm_cfg = load_normalization_cfg()
    overall: dict[str, dict[str, dict[str, Any]]] = {
        s: {sp: _blank_cell() for sp in SPLITS} for s in systems
    }
    vote_bins: dict[str, dict[str, dict[str, Any]]] = {
        name: {sp: _blank_cell() for sp in SPLITS} for name, _, _ in CONF_BINS
    }

    per_utt_path = metrics_dir / "cer_per_utt.jsonl"
    n_rows = 0
    with per_utt_path.open("w", encoding="utf-8") as fout:
        for man in manifest:
            utt_id = man["utt_id"]
            split = str(man.get("split") or "")
            ref_norm = normalize(str(man.get("ref") or ""), norm_cfg)
            ref_len = len(ref_norm)
            for system in systems:
                hyp, conf = _system_hyp(system, indexed[system].get(utt_id))
                distance, cer, empty = utt_distance_cer(hyp, ref_norm)
                rec: dict[str, Any] = {
                    "utt_id": utt_id,
                    "split": split,
                    "system": system,
                    "ref_len": ref_len,
                    "distance": distance,
                    "cer": round(cer, 6),
                }
                if system.startswith("vote"):
                    rec["conf"] = conf
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rows += 1
                _add(overall[system]["All"], distance, ref_len, empty)
                if split in overall[system]:
                    _add(overall[system][split], distance, ref_len, empty)
                if system.startswith("vote"):
                    b = conf_bin(conf)
                    if b is not None:
                        _add(vote_bins[b]["All"], distance, ref_len, empty)
                        if split in vote_bins[b]:
                            _add(vote_bins[b][split], distance, ref_len, empty)

    overall_out = {
        s: {sp: _finalise(overall[s][sp]) for sp in SPLITS} for s in systems
    }
    bins_out = {
        name: {sp: _finalise(vote_bins[name][sp]) for sp in SPLITS}
        for name, _, _ in CONF_BINS
    }
    summary = {
        "n_manifest": len(manifest),
        "n_per_utt_rows": n_rows,
        "systems": list(systems),
        "overall": overall_out,
        "vote_conf": bins_out,
        "note": "micro CER = sum(distance)/sum(ref_len); empty hyp = full deletion",
    }
    (metrics_dir / "cer_overall.json").write_text(
        json.dumps({"systems": overall_out}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "cer_vote_conf.json").write_text(
        json.dumps({"vote_P0": bins_out}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (metrics_dir / "cer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_overall_csv(metrics_dir / "cer_overall.csv", systems, overall_out)
    _write_bins_csv(metrics_dir / "cer_vote_conf.csv", bins_out)
    print(_format_overall(systems, overall_out))
    print()
    print(_format_bins(bins_out))
    print(f"wrote {n_rows} per-utt rows -> {per_utt_path}")
    return summary


def _fmt_cer(val: float | None) -> str:
    return "n/a" if val is None else f"{val:.4f}"


def _write_overall_csv(
    path: Path,
    systems: Iterable[str],
    overall: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fields = ["system", "display"]
    for sp in SPLITS:
        key = sp.lower()
        fields.extend([f"{key}_cer", f"{key}_n", f"{key}_n_empty_hyp"])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for system in systems:
            row: dict[str, Any] = {"system": system, "display": DISPLAY.get(system, system)}
            for sp in SPLITS:
                cell = overall[system][sp]
                key = sp.lower()
                row[f"{key}_cer"] = cell["cer"]
                row[f"{key}_n"] = cell["n"]
                row[f"{key}_n_empty_hyp"] = cell["n_empty_hyp"]
            w.writerow(row)


def _write_bins_csv(path: Path, bins: dict[str, dict[str, dict[str, Any]]]) -> None:
    fields = ["conf_bin"]
    for sp in SPLITS:
        key = sp.lower()
        fields.extend([f"{key}_cer", f"{key}_n", f"{key}_n_empty_hyp"])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, _, _ in CONF_BINS:
            row: dict[str, Any] = {"conf_bin": name}
            for sp in SPLITS:
                cell = bins[name][sp]
                key = sp.lower()
                row[f"{key}_cer"] = cell["cer"]
                row[f"{key}_n"] = cell["n"]
                row[f"{key}_n_empty_hyp"] = cell["n_empty_hyp"]
            w.writerow(row)


def _format_overall(
    systems: Iterable[str],
    overall: dict[str, dict[str, dict[str, Any]]],
) -> str:
    lines = ["=== Overall micro CER ==="]
    header = f"{'system':<12}" + "".join(f"{sp:>22}" for sp in SPLITS)
    sub = f"{'':<12}" + "".join(f"{'cer     n  empty':>22}" for _ in SPLITS)
    lines.append(header)
    lines.append(sub)
    for system in systems:
        name = DISPLAY.get(system, system)
        chunks = [f"{name:<12}"]
        for sp in SPLITS:
            c = overall[system][sp]
            chunks.append(f"{_fmt_cer(c['cer']):>8} {c['n']:>6} {c['n_empty_hyp']:>5}")
        lines.append("".join(chunks))
    return "\n".join(lines)


def _format_bins(bins: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = ["=== vote_P0 micro CER by confidence bin ==="]
    header = f"{'bin':<12}" + "".join(f"{sp:>22}" for sp in SPLITS)
    sub = f"{'':<12}" + "".join(f"{'cer     n  empty':>22}" for _ in SPLITS)
    lines.append(header)
    lines.append(sub)
    for name, _, _ in CONF_BINS:
        chunks = [f"{name:<12}"]
        for sp in SPLITS:
            c = bins[name][sp]
            chunks.append(f"{_fmt_cer(c['cer']):>8} {c['n']:>6} {c['n_empty_hyp']:>5}")
        lines.append("".join(chunks))
    return "\n".join(lines)
