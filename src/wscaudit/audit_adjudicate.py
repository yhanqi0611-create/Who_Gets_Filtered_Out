"""Round-1 audit cleanup, contains-tag merge, and paper-facing metrics."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from wscaudit.asr.base import load_jsonl
from wscaudit.audit import (
    SEED,
    _auto_category,
    merge_audit,
    write_csv,
)
from wscaudit.metrics.markers import DEFAULT_MARKERS, load_markers

CATEGORY_ORDER = (
    "mandarinization",
    "over_dialect",
    "homophone",
    "other",
    "no_error",
    "ref_error",
)
CATEGORY_SET = frozenset(CATEGORY_ORDER)
ERROR_TAGS = frozenset({"mandarinization", "over_dialect", "homophone", "other"})
VARIETY_ORDER = ("sichuanese", "accented_mandarin", "other", "unsure")
VARIETY_SET = frozenset(VARIETY_ORDER)
REF_CORRECT_SET = frozenset({"y", "n", "unsure"})
BOOL_TAG_COLS = tuple(f"h_is_{t}" for t in CATEGORY_ORDER) + ("h_any_error",)
N_CONTROL_DIALECT_DESIGN = 40
UNWEIGHTED_NOTE = "UNWEIGHTED – do not report as recall"


class AuditTagError(ValueError):
    """Unknown or unparseable audit tag."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("audit_id")]


def canonical_tags(raw: str, *, audit_id: str = "") -> list[str]:
    parts = [p.strip().lower() for p in (raw or "").split("|")]
    parts = [p for p in parts if p]
    unknown = [p for p in parts if p not in CATEGORY_SET]
    if unknown:
        raise AuditTagError(f"{audit_id}: unknown h_category tag(s) {unknown} in {raw!r}")
    seen: list[str] = []
    for tag in CATEGORY_ORDER:
        if tag in parts and tag not in seen:
            seen.append(tag)
    return seen


def canonical_category(raw: str, *, audit_id: str = "") -> str:
    return "|".join(canonical_tags(raw, audit_id=audit_id))


def canonical_variety(raw: str, *, audit_id: str = "") -> str:
    parts = [p.strip().lower() for p in (raw or "").split("|")]
    parts = [p for p in parts if p]
    if not parts:
        return "missing"
    unknown = [p for p in parts if p not in VARIETY_SET]
    if unknown:
        raise AuditTagError(f"{audit_id}: unknown h_variety tag(s) {unknown} in {raw!r}")
    seen: list[str] = []
    for tag in VARIETY_ORDER:
        if tag in parts and tag not in seen:
            seen.append(tag)
    return "|".join(seen)


def _canon_choice(raw: str, allowed: frozenset[str], *, empty_as: str = "missing") -> str:
    s = (raw or "").strip().lower()
    if not s:
        return empty_as
    if s == empty_as or s in allowed:
        return s
    raise AuditTagError(f"unknown value {raw!r}")


def only_ref_error(tags: Iterable[str]) -> bool:
    t = [x for x in tags if x]
    return t == ["ref_error"]


def persist_filled(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    dest = dest.resolve()
    if src != dest:
        shutil.copy2(src, dest)
    return dest


def cleanup_human(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    out: list[dict[str, str]] = []
    log: list[dict[str, str]] = []
    n_var_missing = 0
    n_ref_missing = 0
    for r in rows:
        aid = r.get("audit_id") or ""
        row = dict(r)
        old_cat = r.get("h_category") or ""
        tags = canonical_tags(old_cat, audit_id=aid)
        new_cat = "|".join(tags)
        if new_cat != old_cat:
            log.append({"audit_id": aid, "column": "h_category", "old": old_cat, "new": new_cat})
        row["h_category"] = new_cat
        try:
            new_var = canonical_variety(r.get("h_variety") or "", audit_id=aid)
        except AuditTagError as e:
            raise AuditTagError(f"{aid}: h_variety {e}") from e
        old_var = r.get("h_variety") or ""
        if new_var != old_var:
            log.append({"audit_id": aid, "column": "h_variety", "old": old_var, "new": new_var})
        if new_var == "missing":
            n_var_missing += 1
        row["h_variety"] = new_var
        try:
            new_ref = _canon_choice(r.get("h_ref_correct") or "", REF_CORRECT_SET)
        except AuditTagError as e:
            raise AuditTagError(f"{aid}: h_ref_correct {e}") from e
        old_ref = r.get("h_ref_correct") or ""
        if new_ref != old_ref:
            log.append({"audit_id": aid, "column": "h_ref_correct", "old": old_ref, "new": new_ref})
        if new_ref == "missing":
            n_ref_missing += 1
        row["h_ref_correct"] = new_ref
        tagset = set(tags)
        for t in CATEGORY_ORDER:
            row[f"h_is_{t}"] = "1" if t in tagset else "0"
        row["h_any_error"] = "1" if tagset & ERROR_TAGS else "0"
        out.append(row)
    stats = {
        "n": len(out),
        "n_changed_cells": len(log),
        "n_variety_missing": n_var_missing,
        "n_ref_correct_missing": n_ref_missing,
    }
    return out, log, stats


def join_key_human(
    key_rows: list[dict[str, str]],
    human_rows: list[dict[str, str]],
    *,
    expected_n: int | None = None,
) -> list[dict[str, str]]:
    key = {r["audit_id"]: r for r in key_rows}
    human = {r["audit_id"]: r for r in human_rows}
    if len(key) != len(key_rows):
        raise AuditTagError("duplicate audit_id in key")
    if len(human) != len(human_rows):
        raise AuditTagError("duplicate audit_id in human")
    missing = sorted(set(key) - set(human))
    extra = sorted(set(human) - set(key))
    if missing or extra:
        raise AuditTagError(f"unmatched audit_id missing={missing[:8]} extra={extra[:8]}")
    if expected_n is not None and len(key) != expected_n:
        raise AuditTagError(f"expected n={expected_n}, got {len(key)}")
    joined: list[dict[str, str]] = []
    for aid, kr in key.items():
        row = dict(kr)
        row.update(human[aid])
        row["audit_id"] = aid
        row["auto_category"] = (kr.get("auto_category") or "").strip()
        joined.append(row)
    return joined


def _truthy(row: dict[str, str], col: str) -> bool:
    return str(row.get(col) or "").strip() in {"1", "true", "True"}


def _auto_flags(auto: str) -> set[str]:
    a = (auto or "").strip()
    if a == "both":
        return {"mandarinization", "over_dialect"}
    if a in {"mandarinization", "over_dialect"}:
        return {a}
    return set()


def control_frame_n(mark_rows: list[dict[str, Any]]) -> tuple[int, int]:
    """N_ctrl_pop, N_cand_mandarinization (incl. both) on full P0 marker table."""
    n_ctrl = 0
    n_cand_m = 0
    for rec in mark_rows:
        cat = _auto_category(rec)
        if cat in {"mandarinization", "both"}:
            n_cand_m += 1
        if cat:
            continue
        if rec.get("has_dialect_ref"):
            n_ctrl += 1
    return n_ctrl, n_cand_m


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def weighted_recall(
    tp_cand: np.ndarray,
    miss_ctrl: np.ndarray,
    n_ctrl_pop: int,
    n_ctrl_design: int = N_CONTROL_DIALECT_DESIGN,
) -> float:
    w = n_ctrl_pop / n_ctrl_design if n_ctrl_design else float("nan")
    tp = float(np.sum(tp_cand))
    fn = w * float(np.sum(miss_ctrl))
    den = tp + fn
    if den <= 0:
        return float("nan")
    return tp / den


def bootstrap_weighted_recall(
    tp_cand: np.ndarray,
    miss_ctrl: np.ndarray,
    n_ctrl_pop: int,
    *,
    B: int = 10_000,
    seed: int = SEED,
    n_ctrl_design: int = N_CONTROL_DIALECT_DESIGN,
) -> tuple[float, tuple[float, float]]:
    hat = weighted_recall(tp_cand, miss_ctrl, n_ctrl_pop, n_ctrl_design)
    rng = np.random.default_rng(seed)
    n1 = int(tp_cand.size)
    n0 = int(miss_ctrl.size)
    boots = np.empty(B, dtype=float)
    for b in range(B):
        a = tp_cand[rng.integers(0, n1, n1)] if n1 else tp_cand
        c = miss_ctrl[rng.integers(0, n0, n0)] if n0 else miss_ctrl
        boots[b] = weighted_recall(a, c, n_ctrl_pop, n_ctrl_design)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return hat, (float(lo), float(hi))


def contains_confusion(joined: list[dict[str, str]]) -> tuple[list[str], list[str], dict[tuple[str, str], int]]:
    autos = sorted({r.get("auto_category") or "" for r in joined})
    tags = list(CATEGORY_ORDER) + ["any_error"]
    conf: dict[tuple[str, str], int] = {}
    for a in autos:
        for t in tags:
            col = "h_any_error" if t == "any_error" else f"h_is_{t}"
            conf[(a, t)] = sum(
                1 for r in joined if (r.get("auto_category") or "") == a and _truthy(r, col)
            )
    return autos, tags, conf


def format_contains_confusion(
    autos: list[str], tags: list[str], conf: dict[tuple[str, str], int]
) -> str:
    header = ["auto\\\\h_is"] + tags
    lines = ["\t".join(header)]
    for a in autos:
        lines.append("\t".join([a] + [str(conf.get((a, t), 0)) for t in tags]))
    return "\n".join(lines) + "\n"


def compute_metrics(
    joined: list[dict[str, str]],
    *,
    n_ctrl_pop: int,
    B: int = 10_000,
    seed: int = SEED,
) -> dict[str, Any]:
    n_ref_only = sum(1 for r in joined if only_ref_error(canonical_tags(r.get("h_category") or "", audit_id=r.get("audit_id") or "")))

    def precision_for(detector: str) -> dict[str, Any]:
        flagged = [
            r
            for r in joined
            if detector in _auto_flags(r.get("auto_category") or "")
            and not only_ref_error(canonical_tags(r.get("h_category") or "", audit_id=r.get("audit_id") or ""))
        ]
        n = len(flagged)
        k = sum(1 for r in flagged if _truthy(r, f"h_is_{detector}"))
        lo, hi = wilson_ci(k, n)
        return {"detector": detector, "n": n, "k": k, "precision": (k / n if n else float("nan")), "ci": (lo, hi)}

    p_m = precision_for("mandarinization")
    p_o = precision_for("over_dialect")

    cand = [
        r
        for r in joined
        if "mandarinization" in _auto_flags(r.get("auto_category") or "")
        and not only_ref_error(canonical_tags(r.get("h_category") or "", audit_id=r.get("audit_id") or ""))
    ]
    ctrl = [r for r in joined if (r.get("auto_category") or "") == "control_dialect"]
    tp_arr = np.array([1 if _truthy(r, "h_is_mandarinization") else 0 for r in cand], dtype=int)
    miss_arr = np.array([1 if _truthy(r, "h_is_mandarinization") else 0 for r in ctrl], dtype=int)
    if int(tp_arr.sum()) == 0 and int(miss_arr.sum()) == 0:
        rec, rec_ci = float("nan"), (float("nan"), float("nan"))
    else:
        rec, rec_ci = bootstrap_weighted_recall(tp_arr, miss_arr, n_ctrl_pop, B=B, seed=seed)
    tp = int(tp_arr.sum())
    misses_ctrl = int(miss_arr.sum())
    n_human_m = sum(1 for r in joined if _truthy(r, "h_is_mandarinization"))
    naive_den = n_human_m
    naive = (tp / naive_den) if naive_den else float("nan")

    low = [r for r in joined if (r.get("auto_category") or "") == "control_lowconf"]
    k_low = sum(1 for r in low if _truthy(r, "h_any_error"))
    n_low = len(low)
    low_ci = wilson_ci(k_low, n_low)
    low_dist = Counter()
    for r in low:
        for t in CATEGORY_ORDER:
            if _truthy(r, f"h_is_{t}"):
                low_dist[t] += 1

    tag_dist = Counter()
    for r in joined:
        for t in CATEGORY_ORDER:
            if _truthy(r, f"h_is_{t}"):
                tag_dist[t] += 1

    return {
        "n": len(joined),
        "n_ref_error_only": n_ref_only,
        "precision_mandarinization": p_m,
        "precision_over_dialect": p_o,
        "tp_cand": tp,
        "n_cand": int(tp_arr.size),
        "misses_ctrl": misses_ctrl,
        "n_ctrl_sample": int(miss_arr.size),
        "n_ctrl_pop": n_ctrl_pop,
        "weight": n_ctrl_pop / N_CONTROL_DIALECT_DESIGN,
        "recall": rec,
        "recall_ci": rec_ci,
        "naive_unweighted": naive,
        "naive_unweighted_frac": f"{tp}/{naive_den}",
        "lowconf_k": k_low,
        "lowconf_n": n_low,
        "lowconf_rate": (k_low / n_low if n_low else float("nan")),
        "lowconf_ci": low_ci,
        "lowconf_tag_dist": dict(low_dist),
        "tag_dist": dict(tag_dist),
        "over_dialect_human_n": int(tag_dist.get("over_dialect", 0)),
    }


def metrics_csv_rows(m: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("precision_mandarinization", "precision_over_dialect"):
        p = m[key]
        rows.append(
            {
                "metric": f"precision_{p['detector']}",
                "k": p["k"],
                "n": p["n"],
                "estimate": p["precision"],
                "ci_low": p["ci"][0],
                "ci_high": p["ci"][1],
                "note": "Wilson 95%; ref_error-only excluded from denom; both counts for each",
            }
        )
    rows.append(
        {
            "metric": "recall_mandarinization_weighted",
            "k": m["tp_cand"],
            "n": m["n_ctrl_pop"],
            "estimate": m["recall"],
            "ci_low": m["recall_ci"][0],
            "ci_high": m["recall_ci"][1],
            "note": f"TP_cand/(TP_cand+w*misses_ctrl); w=N_ctrl_pop/40={m['weight']:.6g}; N_ctrl_pop={m['n_ctrl_pop']}; misses_ctrl={m['misses_ctrl']}",
        }
    )
    rows.append(
        {
            "metric": "recall_mandarinization_unweighted",
            "k": m["tp_cand"],
            "n": m["naive_unweighted_frac"].split("/")[-1],
            "estimate": m["naive_unweighted"],
            "ci_low": "",
            "ci_high": "",
            "note": UNWEIGHTED_NOTE + f" {m['naive_unweighted_frac']}",
        }
    )
    rows.append(
        {
            "metric": "control_lowconf_error_rate",
            "k": m["lowconf_k"],
            "n": m["lowconf_n"],
            "estimate": m["lowconf_rate"],
            "ci_low": m["lowconf_ci"][0],
            "ci_high": m["lowconf_ci"][1],
            "note": "h_any_error Wilson 95%",
        }
    )
    rows.append(
        {
            "metric": "n_ref_error_only",
            "k": m["n_ref_error_only"],
            "n": m["n"],
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "note": "excluded from detector precision denoms",
        }
    )
    rows.append(
        {
            "metric": "human_over_dialect_n",
            "k": m["over_dialect_human_n"],
            "n": m["n"],
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "note": "multi-label contains over_dialect",
        }
    )
    return rows


def _fmt_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo != lo or hi != hi:  # nan
        return "---"
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def _fmt_pct(x: float) -> str:
    if x != x:
        return "---"
    return f"{100 * x:.1f}"


def write_tex(path: Path, m: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = m["precision_mandarinization"]
    po = m["precision_over_dialect"]
    footnote = (
        "Single annotator, round 1. Contains-tag (multi-label). "
        "Precision denominators exclude rows whose only tag is \\texttt{ref_error} "
        f"(n={m['n_ref_error_only']}). "
        "Mandarinization recall is sampling-weighted: candidates are a census; "
        "control\\_dialect is a sample of $N_{{\\mathrm{{ctrl}}}}$="
        f"{m['n_ctrl_pop']} with weight $N_{{\\mathrm{{ctrl}}}}/40$. "
        "The over\\_dialect detector is unreliable at this precision; "
        "do not treat auto over\\_dialect counts as a result. "
        "Unweighted $23/45$-style recall is not reported."
    )
    lines = [
        r"% Auto-generated by wscaudit audit-merge --write-dir. Do not edit by hand.",
        r"\begin{tabular}{lrrll}",
        r"\toprule",
        r"Detector & $n$ flagged & Human-confirmed & Precision [95\% CI] & Est.\ recall [95\% CI] \\",
        r"\midrule",
        (
            f"Mandarinization & {pm['n']} & {pm['k']} & "
            f"{_fmt_pct(pm['precision'])}\\% {_fmt_ci(pm['ci'])} & "
            f"{_fmt_pct(m['recall'])}\\% {_fmt_ci(m['recall_ci'])} \\\\"
        ),
        (
            f"Over-dialect & {po['n']} & {po['k']} & "
            f"{_fmt_pct(po['precision'])}\\% {_fmt_ci(po['ci'])} & --- \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption*{" + footnote + "}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _hit_list(raw: str) -> list[str]:
    return [p for p in (raw or "").split("|") if p.strip()]


def over_dialect_fp_groups(joined: list[dict[str, str]], n_ex: int = 5) -> list[dict[str, Any]]:
    fps = [
        r
        for r in joined
        if "over_dialect" in _auto_flags(r.get("auto_category") or "")
        and not _truthy(r, "h_is_over_dialect")
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in fps:
        ref_h = set(_hit_list(r.get("ref_dialect_hits") or r.get("ref_dialect") or ""))
        lab_h = _hit_list(r.get("label_dialect_hits") or r.get("hyp_dialect") or "")
        trig = [t for t in lab_h if t not in ref_h] or ["(none in label_dialect_hits)"]
        for form in trig:
            grouped.setdefault(form, []).append(r)
    rows = []
    for form, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        tags = Counter(i.get("h_category") or "" for i in items)
        rows.append(
            {
                "form": form,
                "n": len(items),
                "human_tags": dict(tags),
                "audit_ids": [i["audit_id"] for i in items[:n_ex]],
            }
        )
    return rows


def control_miss_spans(
    joined: list[dict[str, str]],
    lexicon_dialect: set[str],
) -> Counter:
    freq: Counter = Counter()
    misses = [
        r
        for r in joined
        if (r.get("auto_category") or "") == "control_dialect" and _truthy(r, "h_is_mandarinization")
    ]
    for r in misses:
        ref = r.get("ref_norm") or ""
        hyp = r.get("label_vote") or ""
        sm = SequenceMatcher(a=ref, b=hyp, autojunk=False)
        for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
            if tag not in {"replace", "delete"} or i2 <= i1:
                continue
            span = ref[i1:i2]
            if not span:
                continue
            in_lex = span in lexicon_dialect or any(d and d in span for d in lexicon_dialect)
            if not in_lex:
                freq[span] += 1
    return freq


def write_guideline_v2(
    path: Path,
    *,
    fp_groups: list[dict[str, Any]],
    miss_freq: Counter,
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Guideline v2 notes (descriptive; no lexicon or detector changes)",
        "",
        "Round-1 adjudication only. Do **not** retune `markers.tsv` on this sample.",
        "Use these notes when drawing a **new** sample (round 2).",
        "",
        "## Auto over_dialect false positives",
        "",
        "Human `h_is_over_dialect=0` while auto flagged over_dialect (or both). "
        "Grouped by dialect form in the vote label that is absent from the reference hits.",
        "",
    ]
    if not fp_groups:
        lines.append("(none)")
    for g in fp_groups:
        tags = ", ".join(f"{k!r}×{v}" for k, v in g["human_tags"].items())
        ids = ", ".join(g["audit_ids"])
        lines.append(f"- `{g['form']}`: n={g['n']}; human tags: {tags}; examples: {ids}")
    lines += [
        "",
        "## control_dialect misses (not in markers.tsv)",
        "",
        "Utterances auto=`control_dialect` with human mandarinization. "
        "Character-level ref vs vote mismatches whose ref span is **not** a lexicon dialect form. "
        "Frequency table for manual review; do not auto-add to the lexicon.",
        "",
    ]
    if not miss_freq:
        lines.append("(none)")
    for span, n in miss_freq.most_common():
        show = span.replace("\n", " ")
        lines.append(f"- `{show}` × {n}")
    lines += [
        "",
        "## Open schema questions",
        "",
        "- **Multi-tag order:** stored in fixed order "
        "`mandarinization|over_dialect|homophone|other` (`no_error`/`ref_error` exclusive). "
        "`other|homophone` is the same labels as `homophone|other`.",
        "- **homophone vs mandarinization overlap:** a dialect↔Mandarin lexicon pair is mandarinization; "
        "near-homophones that are not that pair (立马/绿马) are homophone. Both may apply on one utterance.",
        "- **ref_error:** valid exclusive `h_category`. Counted separately "
        f"(n={metrics.get('n_ref_error_only', 0)} in round 1) and excluded from detector precision denominators. "
        "`h_ref_correct` remains the independent ref-vs-audio flag.",
        "",
        f"Human contains over_dialect: n={metrics.get('over_dialect_human_n', 0)}. "
        "Auto over_dialect precision is low; treat the detector as unreliable.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _p0_marker_rows(path: Path) -> list[dict[str, Any]]:
    return [r for r in load_jsonl(path) if (r.get("system") or "vote_P0") == "vote_P0"]


def run_adjudication(
    *,
    key_path: Path,
    human_path: Path,
    write_dir: Path,
    markers_jsonl: Path,
    tex_path: Path,
    markers_tsv: Path | None = None,
    expected_n: int | None = 126,
    B: int = 10_000,
    seed: int = SEED,
    stats_table: Path | None = None,
) -> dict[str, Any]:
    write_dir.mkdir(parents=True, exist_ok=True)
    sha_before = sha256_file(stats_table) if stats_table and stats_table.is_file() else None

    filled = persist_filled(human_path, write_dir / "audit_round1_blind_filled.csv")
    key_sha = sha256_file(key_path)
    filled_sha = sha256_file(filled)
    (write_dir / "PROVENANCE.txt").write_text(
        "\n".join(
            [
                f"key={key_path.resolve()}",
                f"key_sha256={key_sha}",
                f"filled={filled.resolve()}",
                f"filled_sha256={filled_sha}",
                f"stats_table_sha256_before={sha_before}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    human_rows = _read_csv(filled)
    clean, log, clean_stats = cleanup_human(human_rows)
    print(f"variety empty→missing: {clean_stats['n_variety_missing']}")
    print(f"h_ref_correct empty→missing: {clean_stats['n_ref_correct_missing']}")
    extra_cols = [c for c in (clean[0] if clean else {}) if c not in human_rows[0]]
    clean_cols = list(human_rows[0].keys()) + [c for c in BOOL_TAG_COLS if c not in human_rows[0]]
    for c in extra_cols:
        if c not in clean_cols:
            clean_cols.append(c)
    write_csv(write_dir / "audit_round1_clean.csv", clean, clean_cols)
    (write_dir / "cleanup_log.txt").write_text(
        "audit_id\tcolumn\told\tnew\n"
        + "".join(f"{x['audit_id']}\t{x['column']}\t{x['old']}\t{x['new']}\n" for x in log),
        encoding="utf-8",
    )
    print(f"cleanup change count: {clean_stats['n_changed_cells']}")

    old = merge_audit(key_path, filled)
    key_rows = _read_csv(key_path)
    joined = join_key_human(key_rows, clean, expected_n=expected_n)
    autos, tags, conf = contains_confusion(joined)
    conf_txt = format_contains_confusion(autos, tags, conf)
    (write_dir / "audit_merge_confusion.txt").write_text(conf_txt, encoding="utf-8")
    conf_csv_rows = [{"auto_category": a, **{t: conf.get((a, t), 0) for t in tags}} for a in autos]
    write_csv(write_dir / "audit_merge_confusion.csv", conf_csv_rows, ["auto_category", *tags])

    mark_rows = _p0_marker_rows(markers_jsonl)
    n_ctrl_pop, n_cand_m = control_frame_n(mark_rows)
    metrics = compute_metrics(joined, n_ctrl_pop=n_ctrl_pop, B=B, seed=seed)
    met_rows = metrics_csv_rows(metrics)
    write_csv(
        write_dir / "audit_metrics.csv",
        met_rows,
        ["metric", "k", "n", "estimate", "ci_low", "ci_high", "note"],
    )
    write_tex(tex_path, metrics)

    lexicon = {p["dialect"] for p in load_markers(markers_tsv or DEFAULT_MARKERS)}
    fps = over_dialect_fp_groups(joined)
    miss_freq = control_miss_spans(joined, lexicon)
    write_guideline_v2(write_dir / "guideline_v2_notes.md", fp_groups=fps, miss_freq=miss_freq, metrics=metrics)

    pm = metrics["precision_mandarinization"]
    po = metrics["precision_over_dialect"]
    print(
        f"precision mandarinization: {pm['k']}/{pm['n']} = {pm['precision']:.4f} "
        f"Wilson {_fmt_ci(pm['ci'])}"
    )
    print(
        f"precision over_dialect: {po['k']}/{po['n']} = {po['precision']:.4f} "
        f"Wilson {_fmt_ci(po['ci'])} (detector unreliable; not a result)"
    )
    print(
        f"weighted recall mandarinization: {metrics['recall']:.4f} "
        f"CI {_fmt_ci(metrics['recall_ci'])} N_ctrl_pop={n_ctrl_pop} N_cand_m={n_cand_m} "
        f"TP={metrics['tp_cand']} misses_ctrl={metrics['misses_ctrl']}"
    )
    print(f"{UNWEIGHTED_NOTE}: {metrics['naive_unweighted_frac']}")
    print(f"n_ref_error_only={metrics['n_ref_error_only']}")

    if sha_before is not None and stats_table is not None and stats_table.is_file():
        sha_after = sha256_file(stats_table)
        if sha_after != sha_before:
            raise RuntimeError("metrics/stats_table.csv changed; abort")
        with (write_dir / "PROVENANCE.txt").open("a", encoding="utf-8") as f:
            f.write(f"stats_table_sha256_after={sha_after}\n")

    return {
        "old_merge": old,
        "clean_stats": clean_stats,
        "metrics": metrics,
        "n_ctrl_pop": n_ctrl_pop,
        "joined_n": len(joined),
    }
