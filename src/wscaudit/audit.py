"""Human-audit sample (round 1) and merge vs auto categories."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from urllib.parse import quote

from wscaudit.asr.base import load_jsonl
from wscaudit.data import HF_REPO_ID, wav_relpath
from wscaudit.normalize import load_normalization_cfg, normalize

SEED = 20261103
N_CONTROL_DIALECT = 40
N_CONTROL_LOW = 20
N_REANNOTATE = 100
HUMAN_COLS = ("h_category", "h_variety", "h_ref_correct", "h_notes")
FULL_COLS = [
    "audit_id",
    "utt_id",
    "split",
    "wav_path",
    "links",
    "dur",
    "conf",
    "cer_vote",
    "ref_norm",
    "label_vote",
    "hyp_sensevoice",
    "hyp_firered",
    "hyp_qwen3asr",
    "ref_dialect_hits",
    "label_dialect_hits",
    "auto_category",
    *HUMAN_COLS,
]
BLIND_DROP = {"auto_category", "conf", "cer_vote"}
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main"


def audio_link(split: str, utt_id: str) -> str:
    """Public clickable wav URL (Hugging Face). Safe for Sheets/Excel."""
    rel = wav_relpath(str(split), str(utt_id))
    encoded = "/".join(quote(part, safe=".-_~") for part in rel.split("/"))
    return f"{HF_RESOLVE}/{encoded}"
GUIDELINES = """# Audit guidelines (round 1)

Listen to the wav. Judge the **vote label against what is said**, not against the written reference.

## Auto categories (shown only in the key file)

These match the automatic detector in `wscaudit.metrics.markers` (candidates, not gold):

- **mandarinization** — a dialect form is in the reference, absent from the label, and the paired Mandarin form is in the label (e.g. ref 啥子 → label 什么).
- **over_dialect** — a dialect form is in the label, absent from the reference, and (if the lexicon has a Mandarin pair) that Mandarin form is in the reference (e.g. ref 什么 → label 啥子). Particles with empty Mandarin still count as over-dialect if inserted.

## Human `h_category`

- **mandarinization** — you hear a dialect form, but the label uses a Mandarin equivalent from the lexicon.
- **over_dialect** — you do **not** hear that dialect form, but the label inserts one (or replaces Mandarin with dialect).
- **homophone** — label and speech differ by a near-homophone that is not the dialect↔Mandarin pair above (e.g. 巴适 vs 八十).
- **other** — some other label error (deletion, insertion, substitution) that is not the above.
- **no_error** — the label matches what was said (even if it disagrees with the reference).
- **ref_error** — the **reference** is wrong relative to the audio; the label may still be fine.

## Human `h_variety`

`sichuanese` | `accented_mandarin` | `other` | `unsure` — what you hear, not what the reference claims.

## Human `h_ref_correct`

`y` | `n` | `unsure` — is the written reference a correct transcript of the audio?

Fill `h_notes` freely. Do not look at `auto_category`, `conf`, or `cer` while annotating (use `audit_round1_blind.csv`).
"""

README_TMPL = """# Audit round 1

Blind file for annotation: `audit_round1_blind.csv` (UTF-8 BOM).
Key (do not send to the annotator first): `audit_round1.csv`.

## Counts

{counts}

## How to merge

Save the filled blind CSV (must keep `audit_id` and human columns `h_*`) and run:

```bash
python -m wscaudit audit-merge --key audit_round1.csv --human audit_round1_blind.csv
```

Prints a confusion table auto_category × h_category (candidates vs controls).

## Re-annotation

`reannotate_ids.txt` lists audit_ids reserved for a ≥3-week intra-annotator pass (κ).
"""


def _join_hits(xs: Any) -> str:
    if not xs:
        return ""
    if isinstance(xs, str):
        return xs
    return "|".join(str(x) for x in xs)


def _auto_category(rec: dict[str, Any]) -> str | None:
    m = int(rec.get("mandarinization_n") or 0)
    o = int(rec.get("over_dialect_n") or 0)
    if m > 0 and o > 0:
        return "both"
    if m > 0:
        return "mandarinization"
    if o > 0:
        return "over_dialect"
    return None


def _index_system(rows: list[dict[str, Any]], system: str | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if "utt_id" not in r:
            continue
        if system is not None:
            sys = r.get("system") or "vote_P0"
            if sys != system:
                continue
        out[r["utt_id"]] = r
    return out


def select_audit_utts(
    mark_p0: list[dict[str, Any]],
    cer_p0: dict[str, dict[str, Any]],
    *,
    seed: int = SEED,
    n_dialect: int = N_CONTROL_DIALECT,
    n_low: int = N_CONTROL_LOW,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return selected marker rows (with auto_category set) and counts."""
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    pool_d: list[dict[str, Any]] = []
    pool_low: list[dict[str, Any]] = []
    cand_ids: set[str] = set()
    for rec in mark_p0:
        uid = rec["utt_id"]
        cat = _auto_category(rec)
        if cat:
            row = dict(rec)
            row["auto_category"] = cat
            candidates.append(row)
            cand_ids.add(uid)
    for rec in mark_p0:
        uid = rec["utt_id"]
        if uid in cand_ids:
            continue
        cer = float((cer_p0.get(uid) or {}).get("cer") or 0.0)
        conf = rec.get("conf")
        conf_f = None if conf is None else float(conf)
        if rec.get("has_dialect_ref"):
            row = dict(rec)
            row["auto_category"] = "control_dialect"
            pool_d.append(row)
        elif (conf_f is not None and conf_f < 0.6) or cer > 0.5:
            row = dict(rec)
            row["auto_category"] = "control_lowconf"
            pool_low.append(row)
    rng.shuffle(pool_d)
    rng.shuffle(pool_low)
    take_d = pool_d[: min(n_dialect, len(pool_d))]
    take_l = pool_low[: min(n_low, len(pool_low))]
    selected = candidates + take_d + take_l
    order = rng.permutation(len(selected))
    ordered = [selected[int(i)] for i in order]
    counts = Counter(r["auto_category"] for r in ordered)
    counts["candidates_total"] = len(candidates)
    counts["total"] = len(ordered)
    return ordered, dict(counts)


def build_audit_rows(
    selected: list[dict[str, Any]],
    *,
    manifest: dict[str, dict[str, Any]],
    vote: dict[str, dict[str, Any]],
    cer_p0: dict[str, dict[str, Any]],
    hyps: dict[str, dict[str, dict[str, Any]]],
    data_root: Path,
) -> list[dict[str, Any]]:
    cfg = load_normalization_cfg()
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(selected, start=1):
        uid = rec["utt_id"]
        man = manifest.get(uid) or {}
        wav = str(man.get("wav") or "")
        wav_path = str(Path(wav) if Path(wav).is_absolute() else (data_root / wav))
        ref = str(man.get("ref") or "")
        rows.append(
            {
                "audit_id": f"A{i:03d}",
                "utt_id": uid,
                "split": str(man.get("split") or rec.get("split") or ""),
                "wav_path": wav_path,
                "links": audio_link(str(man.get("split") or rec.get("split") or ""), uid),
                "dur": man.get("dur"),
                "conf": rec.get("conf"),
                "cer_vote": (cer_p0.get(uid) or {}).get("cer"),
                "ref_norm": normalize(ref, cfg),
                "label_vote": (vote.get(uid) or {}).get("label") or "",
                "hyp_sensevoice": (hyps.get("sensevoice") or {}).get(uid, {}).get("norm") or "",
                "hyp_firered": (hyps.get("firered") or {}).get(uid, {}).get("norm") or "",
                "hyp_qwen3asr": (hyps.get("qwen3asr") or {}).get(uid, {}).get("norm") or "",
                "ref_dialect_hits": _join_hits(rec.get("ref_dialect")),
                "label_dialect_hits": _join_hits(rec.get("hyp_dialect")),
                "auto_category": rec["auto_category"],
                "h_category": "",
                "h_variety": "",
                "h_ref_correct": "",
                "h_notes": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def run_audit(
    out_root: Path,
    data_root: Path,
    *,
    seed: int = SEED,
    n_dialect: int = N_CONTROL_DIALECT,
    n_low: int = N_CONTROL_LOW,
    n_reannotate: int = N_REANNOTATE,
) -> dict[str, Any]:
    audit_dir = out_root / "audit"
    mark_rows = [
        r
        for r in load_jsonl(out_root / "metrics" / "markers_per_utt.jsonl")
        if (r.get("system") or "vote_P0") == "vote_P0"
    ]
    cer_p0 = _index_system(load_jsonl(out_root / "metrics" / "cer_per_utt.jsonl"), "vote_P0")
    selected, counts = select_audit_utts(mark_rows, cer_p0, seed=seed, n_dialect=n_dialect, n_low=n_low)
    manifest = _index_system(load_jsonl(out_root / "manifest.jsonl"))
    vote = _index_system(load_jsonl(out_root / "merged" / "vote_P0.jsonl"))
    hyps = {
        name: _index_system(load_jsonl(out_root / "hyps" / f"{name}.jsonl"))
        for name in ("sensevoice", "firered", "qwen3asr")
    }
    rows = build_audit_rows(
        selected,
        manifest=manifest,
        vote=vote,
        cer_p0=cer_p0,
        hyps=hyps,
        data_root=data_root,
    )
    write_csv(audit_dir / "audit_round1.csv", rows, FULL_COLS)
    blind_cols = [c for c in FULL_COLS if c not in BLIND_DROP]
    write_csv(audit_dir / "audit_round1_blind.csv", rows, blind_cols)
    rng = np.random.default_rng(seed)
    ids = [r["audit_id"] for r in rows]
    rng.shuffle(ids)
    take = ids[: min(n_reannotate, len(ids))]
    (audit_dir / "reannotate_ids.txt").write_text("\n".join(take) + ("\n" if take else ""), encoding="utf-8")
    (audit_dir / "GUIDELINES.md").write_text(GUIDELINES, encoding="utf-8")
    count_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(counts.items()))
    (audit_dir / "README.md").write_text(README_TMPL.format(counts=count_lines), encoding="utf-8")
    print("audit counts:", json.dumps(counts, ensure_ascii=False))
    print(f"wrote {len(rows)} rows -> {audit_dir / 'audit_round1.csv'}")
    print(f"blind -> {audit_dir / 'audit_round1_blind.csv'}")
    return {"n": len(rows), "counts": counts, "dir": str(audit_dir)}


def _insert_links_field(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c != "links"]
    if "wav_path" in cols:
        i = cols.index("wav_path") + 1
        cols.insert(i, "links")
    else:
        cols.append("links")
    return cols


def patch_audit_links(audit_dir: Path) -> dict[str, int]:
    """Add/refresh ``links`` on existing CSVs without reshuffling audit_id."""
    n: dict[str, int] = {}
    for name in ("audit_round1.csv", "audit_round1_blind.csv"):
        path = audit_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            rows = list(reader)
        cols = _insert_links_field(old_fields)
        for r in rows:
            r["links"] = audio_link(str(r.get("split") or ""), str(r.get("utt_id") or ""))
        write_csv(path, rows, cols)
        n[name] = len(rows)
        print(f"patched links on {len(rows)} rows -> {path}")
    if not n:
        raise FileNotFoundError(f"no audit CSVs under {audit_dir}")
    return n


def confusion(auto: list[str], human: list[str]) -> dict[tuple[str, str], int]:
    c: dict[tuple[str, str], int] = Counter(zip(auto, human))
    return dict(c)


def merge_audit(key_path: Path, human_path: Path) -> dict[str, Any]:
    def _read(path: Path) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8-sig") as f:
            return {r["audit_id"]: r for r in csv.DictReader(f) if r.get("audit_id")}

    key = _read(key_path)
    human = _read(human_path)
    missing = sorted(set(key) - set(human))
    extra = sorted(set(human) - set(key))
    autos, hums = [], []
    for aid, kr in key.items():
        hr = human.get(aid) or {}
        a = (kr.get("auto_category") or "").strip()
        h = (hr.get("h_category") or "").strip()
        if a and h:
            autos.append(a)
            hums.append(h)
    conf = confusion(autos, hums)
    labels_a = sorted({a for a, _ in conf})
    labels_h = sorted({h for _, h in conf})
    print(f"merged n_key={len(key)} n_human={len(human)} n_both_labelled={len(autos)}")
    if missing:
        print(f"missing in human ({len(missing)}): {missing[:8]}")
    if extra:
        print(f"extra in human ({len(extra)}): {extra[:8]}")
    header = ["auto\\\\human"] + labels_h
    print("\t".join(header))
    for a in labels_a:
        print("\t".join([a] + [str(conf.get((a, h), 0)) for h in labels_h]))
    return {"n": len(autos), "confusion": {f"{a}|{h}": n for (a, h), n in conf.items()}}
