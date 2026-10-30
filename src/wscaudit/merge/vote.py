"""Character-level majority vote (primary label; no LLM).

Alignment: rapidfuzz Levenshtein opcodes (match 0, sub/ins/del 1) on Unicode
characters, then a 3-sequence centre-star merge. Gaps are None; we never
invent symbols. Ties at a column: SenseVoice, then FireRed, then Qwen3.
Empty/missing ``norm`` is ABSENT, not an empty string.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from wscaudit.asr.base import existing_utt_ids, load_jsonl

GAP = None
DEFAULT_HYPS = ("sensevoice", "firered", "qwen3asr")
TIE_ORDER = ("sensevoice", "firered", "qwen3asr")


def present_norm(rec: dict[str, Any] | None) -> str | None:
    if rec is None:
        return None
    text = rec.get("norm")
    if text is None:
        return None
    s = str(text)
    if not s.strip():
        return None
    return s


def align_pair(a: str, b: str) -> tuple[list[str | None], list[str | None]]:
    """Pairwise alignment with equal sub/ins/del cost (Levenshtein)."""
    a_al: list[str | None] = []
    b_al: list[str | None] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(a, b):
        if tag in {"equal", "replace"}:
            a_al.extend(a[i1:i2])
            b_al.extend(b[j1:j2])
        elif tag == "delete":
            a_al.extend(a[i1:i2])
            b_al.extend([GAP] * (i2 - i1))
        elif tag == "insert":
            a_al.extend([GAP] * (j2 - j1))
            b_al.extend(b[j1:j2])
    return a_al, b_al


def _pick_center(named: list[tuple[str, str]]) -> int:
    best_i = 0
    best_sum = None
    for i, (_, si) in enumerate(named):
        total = sum(Levenshtein.distance(si, sj) for j, (_, sj) in enumerate(named) if i != j)
        if best_sum is None or total < best_sum:
            best_sum = total
            best_i = i
    return best_i


def _merge_other(
    rows: dict[str, list[str | None]],
    center: str,
    c_al: list[str | None],
    s_al: list[str | None],
    new_name: str,
) -> dict[str, list[str | None]]:
    old_c = rows[center]
    out: dict[str, list[str | None]] = {k: [] for k in rows}
    out[new_name] = []
    i = j = 0
    while i < len(old_c) or j < len(c_al):
        old_gap = i < len(old_c) and old_c[i] is GAP
        new_gap = j < len(c_al) and c_al[j] is GAP
        if i < len(old_c) and j < len(c_al) and not old_gap and not new_gap:
            for k in rows:
                out[k].append(rows[k][i])
            out[new_name].append(s_al[j])
            i += 1
            j += 1
        elif i < len(old_c) and old_gap:
            for k in rows:
                out[k].append(rows[k][i])
            out[new_name].append(GAP)
            i += 1
        elif j < len(c_al) and new_gap:
            for k in rows:
                out[k].append(GAP)
            out[new_name].append(s_al[j])
            j += 1
        elif i >= len(old_c):
            for k in rows:
                out[k].append(GAP)
            out[new_name].append(s_al[j] if j < len(c_al) else GAP)
            j += 1
        else:
            for k in rows:
                out[k].append(rows[k][i])
            out[new_name].append(GAP)
            i += 1
    return out


def multiple_align(named: list[tuple[str, str]]) -> dict[str, list[str | None]]:
    """Centre-star MSA. ``named`` is ``[(source, nonempty_norm), ...]``."""
    if not named:
        return {}
    if len(named) == 1:
        name, text = named[0]
        return {name: list(text)}
    if len(named) == 2:
        (n0, s0), (n1, s1) = named
        a0, a1 = align_pair(s0, s1)
        return {n0: a0, n1: a1}
    cidx = _pick_center(named)
    cname, cstr = named[cidx]
    rows: dict[str, list[str | None]] = {cname: list(cstr)}
    for name, text in named:
        if name == cname:
            continue
        c_al, s_al = align_pair(cstr, text)
        rows = _merge_other(rows, cname, c_al, s_al, name)
    return rows


def vote_columns(
    rows: dict[str, list[str | None]],
    tie_order: tuple[str, ...] = TIE_ORDER,
) -> tuple[str, float, float]:
    """Return (label, sentence_conf, n_voters_mean)."""
    if not rows:
        return "", 0.0, 0.0
    names = list(rows)
    n_cols = len(next(iter(rows.values())))
    letters: list[str] = []
    confs: list[float] = []
    n_present_list: list[int] = []
    for col in range(n_cols):
        present: list[tuple[str, str]] = []
        for name in names:
            ch = rows[name][col]
            if ch is not GAP:
                present.append((name, ch))
        if not present:
            continue
        counts = Counter(ch for _, ch in present)
        top = max(counts.values())
        winners = {ch for ch, n in counts.items() if n == top}
        chosen: str | None = None
        if len(winners) == 1:
            chosen = next(iter(winners))
        else:
            by_src = {name: ch for name, ch in present}
            for src in tie_order:
                if src in by_src and by_src[src] in winners:
                    chosen = by_src[src]
                    break
            if chosen is None:
                chosen = present[0][1]
        letters.append(chosen)
        n_here = len(present)
        n_present_list.append(n_here)
        confs.append(counts[chosen] / n_here)
    if not letters:
        return "", 0.0, 0.0
    return (
        "".join(letters),
        sum(confs) / len(confs),
        sum(n_present_list) / len(n_present_list),
    )


def vote_one(
    hyps: dict[str, str | None],
    tie_order: tuple[str, ...] = TIE_ORDER,
) -> dict[str, Any]:
    named = [(n, t) for n, t in hyps.items() if t is not None]
    n_available = len(named)
    sources_used = [n for n, _ in named]
    if n_available == 0:
        return {
            "label": "",
            "conf": 0.0,
            "n_voters_mean": 0.0,
            "n_available": 0,
            "sources_used": [],
            "char_count": 0,
        }
    rows = multiple_align(named)
    label, conf, n_vm = vote_columns(rows, tie_order=tie_order)
    return {
        "label": label,
        "conf": conf,
        "n_voters_mean": n_vm,
        "n_available": n_available,
        "sources_used": sources_used,
        "char_count": len(label),
    }


def _index_hyps(path: Path) -> dict[str, dict[str, Any]]:
    return {row["utt_id"]: row for row in load_jsonl(path) if "utt_id" in row}


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_with = {0: 0, 1: 0, 2: 0, 3: 0}
    by_split: dict[str, list[float]] = {}
    for row in rows:
        k = int(row.get("n_available") or 0)
        if k in n_with:
            n_with[k] += 1
        else:
            n_with[k] = n_with.get(k, 0) + 1
        spl = str(row.get("split") or "")
        by_split.setdefault(spl, []).append(float(row.get("conf") or 0.0))
    mean_conf = {
        spl: (sum(vs) / len(vs) if vs else 0.0) for spl, vs in sorted(by_split.items())
    }
    return {
        "n": len(rows),
        "n_with_0": n_with.get(0, 0),
        "n_with_1": n_with.get(1, 0),
        "n_with_2": n_with.get(2, 0),
        "n_with_3": n_with.get(3, 0),
        "mean_conf_by_split": mean_conf,
        "mean_conf": (sum(float(r.get("conf") or 0.0) for r in rows) / len(rows) if rows else 0.0),
    }


def run_vote(
    manifest: Path,
    hyp_paths: dict[str, Path],
    out_path: Path,
    summary_path: Path,
    *,
    split: str = "all",
    limit: int | None = None,
    hyp_order: tuple[str, ...] = DEFAULT_HYPS,
) -> Path:
    man_rows = load_jsonl(manifest)
    if not man_rows:
        raise FileNotFoundError(f"empty or missing manifest: {manifest}")
    tables = {name: _index_hyps(path) for name, path in hyp_paths.items()}
    skip = existing_utt_ids(out_path)
    split_l = split.lower()
    todo: list[dict[str, Any]] = []
    for row in man_rows:
        uid = row["utt_id"]
        if uid in skip:
            continue
        if split_l not in {"all", ""} and str(row.get("split", "")).lower() != split_l:
            continue
        todo.append(row)
        if limit is not None and len(todo) >= limit:
            break

    print(
        f"vote: split={split} todo={len(todo)} skipped={len(skip)} "
        f"hyps={','.join(hyp_order)} -> {out_path}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for row in tqdm(todo, desc="vote", leave=True):
            t0 = time.perf_counter()
            uid = row["utt_id"]
            hyps: dict[str, str | None] = {}
            for name in hyp_order:
                hyps[name] = present_norm(tables.get(name, {}).get(uid))
            voted = vote_one(hyps, tie_order=TIE_ORDER)
            rec = {
                "utt_id": uid,
                "split": row.get("split"),
                "label": voted["label"],
                "conf": round(float(voted["conf"]), 6),
                "n_voters_mean": round(float(voted["n_voters_mean"]), 6),
                "n_available": voted["n_available"],
                "sources_used": voted["sources_used"],
                "char_count": voted["char_count"],
                "runtime_s": round(time.perf_counter() - t0, 6),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                fout.flush()
        fout.flush()

    all_rows = load_jsonl(out_path)
    summary = _summary_from_rows(all_rows)
    summary["n_written_this_run"] = n_written
    summary["n_skipped"] = len(skip)
    summary["out_path"] = str(out_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"vote: wrote={n_written} n_file={summary['n']} "
        f"n_with_3={summary['n_with_3']} n_with_2={summary['n_with_2']} "
        f"n_with_1={summary['n_with_1']} n_with_0={summary['n_with_0']} "
        f"mean_conf={summary['mean_conf']:.4f} -> {summary_path}"
    )
    return out_path
