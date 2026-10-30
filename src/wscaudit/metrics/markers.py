"""Dialect-marker detection, retention, and confidence pass-rate (RQ1/RQ3).

Auto Mandarinization / over-dialect counts are **candidates for a human error
taxonomy**, not gold labels. Matching is longest-first exact string scan on
already-normalised continuous Chinese (no spaces). One-character forms
(莫 / 幺 / 嘿 and any other len-1 entry) do not match when the previous
character is CJK, so they cannot fire inside 西瓜-style compounds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from wscaudit.asr.base import load_jsonl
from wscaudit.normalize import load_normalization_cfg, normalize

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARKERS = _REPO_ROOT / "resources" / "markers.tsv"
SPLITS = ("Easy", "Hard", "All")
SUMMARY_KEYS = frozenset(
    {
        "lexicon_n",
        "n",
        "has_dialect_ref",
        "token_retention",
        "strict_retention",
        "mandarinization_n",
        "over_dialect_n",
        "pass_0.9",
        "pass_0.6",
        "note",
    }
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF
        or 0x4E00 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2CEAF
    )


def load_markers(path: Path | None = None) -> list[dict[str, str]]:
    """Load TSV with columns dialect, mandarin, note. Skip # comments."""
    path = path or DEFAULT_MARKERS
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        header: list[str] | None = None
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.split("\t")
            if header is None:
                header = [c.strip() for c in cols]
                continue
            rec = {header[i]: (cols[i].strip() if i < len(cols) else "") for i in range(len(header))}
            dialect = rec.get("dialect") or rec.get("form") or ""
            if not dialect:
                continue
            rows.append(
                {
                    "dialect": dialect,
                    "mandarin": rec.get("mandarin") or rec.get("mandarin_equivalents") or "",
                    "note": rec.get("note") or rec.get("notes") or "",
                }
            )
    return rows


def _particle_surfaces(pairs: Iterable[dict[str, str]]) -> set[str]:
    out: set[str] = set()
    for p in pairs:
        note = (p.get("note") or "").lower()
        empty_m = not (p.get("mandarin") or "").strip()
        if empty_m and ("particle" in note or "语气" in (p.get("note") or "")):
            out.add(p["dialect"])
    return out


def _scan_entries(pairs: Iterable[dict[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[str, str]] = []
    for p in pairs:
        d = p["dialect"]
        if d and (d, "dialect") not in seen:
            seen.add((d, "dialect"))
            entries.append((d, "dialect"))
        m = p.get("mandarin") or ""
        if m and (m, "mandarin") not in seen:
            seen.add((m, "mandarin"))
            entries.append((m, "mandarin"))
    entries.sort(key=lambda x: (-len(x[0]), 0 if x[1] == "dialect" else 1, x[0]))
    return entries


def _dialect_mandarin_map(pairs: Iterable[dict[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in pairs:
        d = p["dialect"]
        m = (p.get("mandarin") or "").strip()
        out.setdefault(d, [])
        if m and m not in out[d]:
            out[d].append(m)
    return out


def _ok_at(text: str, i: int, surface: str, particles: set[str]) -> bool:
    if not surface or not text.startswith(surface, i):
        return False
    # Len-1 content (莫/幺/嘿/瓜) must not match as an infix; particles (噻) may.
    if (
        len(surface) == 1
        and surface not in particles
        and i > 0
        and _is_cjk(text[i - 1])
    ):
        return False
    return True


def find_markers(
    text: str,
    pairs: Iterable[dict[str, str]] | None = None,
    scan: list[tuple[str, str]] | None = None,
    particles: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Longest-first non-overlapping exact matches. ``kind`` is dialect|mandarin."""
    pair_list = list(pairs) if pairs is not None else (None if scan is not None else load_markers())
    if pair_list is not None:
        if scan is None:
            scan = _scan_entries(pair_list)
        if particles is None:
            particles = _particle_surfaces(pair_list)
    scan = scan or []
    particles = particles or set()
    text = text or ""
    hits: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        found = False
        for surface, kind in scan:
            if _ok_at(text, i, surface, particles):
                hits.append({"surface": surface, "kind": kind, "start": i, "end": i + len(surface)})
                i += len(surface)
                found = True
                break
        if not found:
            i += 1
    return hits


def _surfaces(hits: list[dict[str, Any]], kind: str) -> list[str]:
    return [h["surface"] for h in hits if h["kind"] == kind]


def _present(needle: str, haystack: str, particles: set[str]) -> bool:
    if not needle:
        return False
    if len(needle) == 1:
        for i, ch in enumerate(haystack):
            if ch == needle and _ok_at(haystack, i, needle, particles):
                return True
        return False
    return needle in haystack


def score_utt(
    ref: str,
    hyp: str,
    pairs: list[dict[str, str]],
    scan: list[tuple[str, str]],
    d2m: dict[str, list[str]],
) -> dict[str, Any]:
    particles = _particle_surfaces(pairs)
    ref_hits = find_markers(ref, pairs=pairs, scan=scan, particles=particles)
    hyp_hits = find_markers(hyp, pairs=pairs, scan=scan, particles=particles)
    ref_d = _surfaces(ref_hits, "dialect")
    hyp_d = _surfaces(hyp_hits, "dialect")
    ref_c = Counter(ref_d)
    hyp_c = Counter(hyp_d)

    mandarinization_n = 0
    for surface in ref_d:
        mandarins = d2m.get(surface) or []
        if not mandarins:
            continue
        if (not _present(surface, hyp, particles)) and any(
            _present(m, hyp, particles) for m in mandarins
        ):
            mandarinization_n += 1

    over_dialect_n = 0
    for surface in hyp_d:
        mandarins = d2m.get(surface) or []
        if _present(surface, ref, particles):
            continue
        if not mandarins:
            over_dialect_n += 1
        elif any(_present(m, ref, particles) for m in mandarins):
            over_dialect_n += 1

    n_ref = len(ref_d)
    if n_ref == 0:
        retained_token_frac = None
        strict_ok = None
    else:
        kept = sum(min(ref_c[s], hyp_c[s]) for s in ref_c)
        retained_token_frac = kept / n_ref
        strict_ok = all(hyp_c[s] >= ref_c[s] for s in ref_c)

    return {
        "ref_hits": ref_hits,
        "hyp_hits": hyp_hits,
        "ref_dialect": ref_d,
        "hyp_dialect": hyp_d,
        "mandarinization_n": mandarinization_n,
        "over_dialect_n": over_dialect_n,
        "retained_token_frac": retained_token_frac,
        "has_dialect_ref": n_ref >= 1,
        "strict_ok": strict_ok,
        "n_ref_dialect": n_ref,
        "n_retained": None if n_ref == 0 else sum(min(ref_c[s], hyp_c[s]) for s in ref_c),
    }


def _blank_ret() -> dict[str, Any]:
    return {"ref_tokens": 0, "retained": 0, "rate": None, "n_utt": 0, "n_strict_ok": 0, "strict_rate": None}


def _blank_pass() -> dict[str, Any]:
    return {
        "has_dialect_ref": {"n": 0, "n_pass": 0, "rate": None},
        "no_dialect_ref": {"n": 0, "n_pass": 0, "rate": None},
        "delta_pp": None,
    }


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def _final_pass(block: dict[str, Any]) -> dict[str, Any]:
    for key in ("has_dialect_ref", "no_dialect_ref"):
        cell = block[key]
        cell["rate"] = _rate(cell["n_pass"], cell["n"])
        if cell["rate"] is not None:
            cell["rate"] = round(cell["rate"], 6)
    rh, rn = block["has_dialect_ref"]["rate"], block["no_dialect_ref"]["rate"]
    block["delta_pp"] = None if rh is None or rn is None else round((rh - rn) * 100, 4)
    return block


def run_markers(
    manifest_path: Path,
    vote_path: Path,
    markers_path: Path | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    pairs = load_markers(markers_path or DEFAULT_MARKERS)
    scan = _scan_entries(pairs)
    d2m = _dialect_mandarin_map(pairs)
    manifest = load_jsonl(manifest_path)
    votes = {r["utt_id"]: r for r in load_jsonl(vote_path) if "utt_id" in r}
    if not manifest:
        raise FileNotFoundError(f"empty or missing manifest: {manifest_path}")
    if not vote_path.is_file():
        raise FileNotFoundError(vote_path)

    metrics_dir = metrics_dir or vote_path.parent.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    norm_cfg = load_normalization_cfg()
    has_ref = {sp: 0 for sp in SPLITS}
    n_split = {sp: 0 for sp in SPLITS}
    tok = {sp: _blank_ret() for sp in SPLITS}
    mand = {sp: 0 for sp in SPLITS}
    over = {sp: 0 for sp in SPLITS}
    p09 = _blank_pass()
    p06 = _blank_pass()

    out_path = metrics_dir / "markers_per_utt.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for man in manifest:
            utt_id = man["utt_id"]
            split = str(man.get("split") or "")
            ref = normalize(str(man.get("ref") or ""), norm_cfg)
            vote = votes.get(utt_id) or {}
            hyp = str(vote.get("label") or "")
            conf_raw = vote.get("conf")
            conf = None if conf_raw is None else float(conf_raw)
            scored = score_utt(ref, hyp, pairs, scan, d2m)
            pass_09 = conf is not None and conf >= 0.9
            pass_06 = conf is not None and conf >= 0.6
            rec = {
                "utt_id": utt_id,
                "split": split,
                "conf": conf,
                "ref_dialect": scored["ref_dialect"],
                "hyp_dialect": scored["hyp_dialect"],
                "mandarinization_n": scored["mandarinization_n"],
                "over_dialect_n": scored["over_dialect_n"],
                "retained_token_frac": scored["retained_token_frac"],
                "has_dialect_ref": scored["has_dialect_ref"],
                "pass_0.9": pass_09,
                "pass_0.6": pass_06,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

            splits_here = ["All"] + ([split] if split in ("Easy", "Hard") else [])
            for sp in splits_here:
                n_split[sp] += 1
                if scored["has_dialect_ref"]:
                    has_ref[sp] += 1
                    tok[sp]["n_utt"] += 1
                    tok[sp]["ref_tokens"] += scored["n_ref_dialect"]
                    tok[sp]["retained"] += scored["n_retained"] or 0
                    if scored["strict_ok"]:
                        tok[sp]["n_strict_ok"] += 1
                mand[sp] += scored["mandarinization_n"]
                over[sp] += scored["over_dialect_n"]

            group = "has_dialect_ref" if scored["has_dialect_ref"] else "no_dialect_ref"
            p09[group]["n"] += 1
            p06[group]["n"] += 1
            if pass_09:
                p09[group]["n_pass"] += 1
            if pass_06:
                p06[group]["n_pass"] += 1

    def _fin_tok(cell: dict[str, Any]) -> dict[str, Any]:
        out = dict(cell)
        out["rate"] = None if out["ref_tokens"] <= 0 else round(out["retained"] / out["ref_tokens"], 6)
        out["strict_rate"] = None if out["n_utt"] <= 0 else round(out["n_strict_ok"] / out["n_utt"], 6)
        return out

    summary = {
        "lexicon_n": len(pairs),
        "n": n,
        "has_dialect_ref": {
            sp: {
                "n_utt": has_ref[sp],
                "n_split": n_split[sp],
                "share": None if n_split[sp] <= 0 else round(has_ref[sp] / n_split[sp], 6),
            }
            for sp in SPLITS
        },
        "token_retention": {sp: _fin_tok(tok[sp]) for sp in SPLITS},
        "strict_retention": {
            sp: {
                "n_eligible": tok[sp]["n_utt"],
                "n_ok": tok[sp]["n_strict_ok"],
                "rate": None if tok[sp]["n_utt"] <= 0 else round(tok[sp]["n_strict_ok"] / tok[sp]["n_utt"], 6),
            }
            for sp in SPLITS
        },
        "mandarinization_n": dict(mand),
        "over_dialect_n": dict(over),
        "pass_0.9": _final_pass(p09),
        "pass_0.6": _final_pass(p06),
        "note": (
            "mandarinization_n / over_dialect_n are automatic candidates for human "
            "error-taxonomy check, not gold. Matching is longest-first; len-1 forms "
            "require a non-CJK previous character."
        ),
    }
    sum_path = metrics_dir / "markers_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {n} rows -> {out_path}")
    print(f"wrote summary -> {sum_path}")
    return summary
