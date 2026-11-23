"""Read-only paper-facing write-up of the round-1 audit (Prompt 16)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from wscaudit.audit_adjudicate import join_key_human, sha256_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_B = 10_000
DEFAULT_SEED = 20261103
DESIGN_N_CTRL = 40
FILTER_THR = "0.9"
WILSON_PCT = 95

_NUM_TOKEN = re.compile(r"\d+\.\d+|\d+")


class Trace:
    """Collect every formatted number with its source."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, formatted: str, *, file: str, row_key: str, column: str, raw_value: Any) -> str:
        self.records.append(
            {
                "file": file,
                "row_key": row_key,
                "column": column,
                "raw_value": raw_value if isinstance(raw_value, str) else _jsonable(raw_value),
                "formatted": formatted,
            }
        )
        return formatted

    def pct(self, x: float | str, *, file: str, row_key: str, column: str) -> str:
        v = float(x)
        return self.add(f"{100.0 * v:.1f}", file=file, row_key=row_key, column=column, raw_value=x)

    def ci_pct(self, lo: float | str, hi: float | str, *, file: str, row_key: str) -> str:
        a = self.pct(lo, file=file, row_key=row_key, column="ci_low")
        b = self.pct(hi, file=file, row_key=row_key, column="ci_high")
        return f"[{a}, {b}]"

    def count(self, x: Any, *, file: str, row_key: str, column: str) -> str:
        n = int(float(x))
        return self.add(str(n), file=file, row_key=row_key, column=column, raw_value=x)

    def sig(self, x: Any, *, file: str, row_key: str, column: str) -> str:
        v = float(x)
        s = f"{v:.3g}"
        return self.add(s, file=file, row_key=row_key, column=column, raw_value=x)

    def wilson_level(self) -> str:
        n = self.add(str(WILSON_PCT), file="code", row_key="wilson", column="percent", raw_value=WILSON_PCT)
        return f"Wilson {n}% CI"


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metrics_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r["metric"]: r for r in rows if r.get("metric")}


def stats_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["name"], r["split"]): r for r in rows if r.get("name")}


def parse_weighted_note(note: str) -> tuple[int, int, int]:
    """Return N_ctrl_pop, misses_ctrl, design n from the metrics note field."""
    pop_m = re.search(r"N_ctrl_pop=([0-9]+)", note or "")
    miss_m = re.search(r"misses_ctrl=([0-9]+)", note or "")
    den_m = re.search(r"N_ctrl_pop/([0-9]+)", note or "")
    if not pop_m or not miss_m:
        raise ValueError(f"cannot parse N_ctrl_pop/misses_ctrl from note: {note!r}")
    design = int(den_m.group(1)) if den_m else DESIGN_N_CTRL
    return int(pop_m.group(1)), int(miss_m.group(1)), design


def parse_b_seed(provenance: str) -> tuple[int, int]:
    b_m = re.search(r"\bB=([0-9]+)", provenance)
    s_m = re.search(r"\bseed=([0-9]+)", provenance)
    b = int(b_m.group(1)) if b_m else DEFAULT_B
    seed = int(s_m.group(1)) if s_m else DEFAULT_SEED
    return b, seed


def delta_pass_word(estimate: float) -> str:
    return "more" if estimate > 0 else "less"


def holm_significant(p_holm: float) -> bool:
    return p_holm < 0.05


def holm_clause(p_pp: float, p_or: float) -> str:
    if not holm_significant(p_pp) and not holm_significant(p_or):
        return "neither is significant after Holm"
    if holm_significant(p_pp) and holm_significant(p_or):
        return "both remain significant after Holm"
    if holm_significant(p_pp):
        return "only the unadjusted difference is significant after Holm"
    return "only the adjusted odds ratio is significant after Holm"


def strata_counts(key_rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter((r.get("auto_category") or "").strip() for r in key_rows if r.get("audit_id")))


def _truthy(row: dict[str, str], col: str) -> bool:
    return str(row.get(col) or "").strip() in {"1", "true", "True"}


def pick_examples(joined: list[dict[str, str]], tags: list[str], *, max_total: int = 5, per_tag: int = 2) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    seen: set[str] = set()

    def ranked(tag: str) -> list[dict[str, str]]:
        cands = [r for r in joined if _truthy(r, f"h_is_{tag}")]
        cands.sort(key=lambda r: (0 if (r.get("h_notes") or "").strip() else 1, r.get("audit_id") or ""))
        return cands

    for tag in tags:
        for r in ranked(tag):
            aid = r.get("audit_id") or ""
            if aid in seen:
                continue
            seen.add(aid)
            chosen.append(r)
            break
        if len(chosen) >= max_total:
            return chosen
    if len(chosen) < max_total:
        for tag in tags:
            n_this = sum(1 for r in chosen if _truthy(r, f"h_is_{tag}"))
            if n_this >= per_tag:
                continue
            for r in ranked(tag):
                aid = r.get("audit_id") or ""
                if aid in seen:
                    continue
                seen.add(aid)
                chosen.append(r)
                break
            if len(chosen) >= max_total:
                break
    return chosen[:max_total]


def dominant_fp_forms(notes: str, *, limit: int = 5) -> list[str]:
    in_sec = False
    items: list[tuple[int, str]] = []
    for line in notes.splitlines():
        if line.startswith("## "):
            in_sec = "over_dialect false" in line.lower()
            continue
        if not in_sec:
            continue
        m = re.match(r"- `([^`]+)`:\s*n=([0-9]+)", line.strip())
        if m:
            items.append((int(m.group(2)), m.group(1)))
    items.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in items[:limit]]


def tag_distribution(rows: list[dict[str, str]], tags: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tags:
        out[t] = sum(1 for r in rows if _truthy(r, f"h_is_{t}"))
    return out


def git_head(repo: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _lowconf_mostly_nondialect(dist: dict[str, int]) -> bool:
    dialectish = dist.get("mandarinization", 0) + dist.get("over_dialect", 0)
    otherish = dist.get("homophone", 0) + dist.get("other", 0)
    return otherish > dialectish


def render_findings(
    *,
    tr: Trace,
    met: dict[str, dict[str, str]],
    stats: dict[tuple[str, str], dict[str, str]],
    strata: dict[str, int],
    n_sample: int,
    n_ref_only: int,
    n_pop: int,
    misses_ctrl: int,
    design_n: int,
    b: int,
    seed: int,
    fp_forms: list[str],
    examples: list[dict[str, str]],
    low_rows: list[dict[str, str]] | None,
    low_dist: dict[str, int] | None,
    include_low: bool,
) -> str:
    n_s = tr.count(n_sample, file="audit_round1.csv", row_key="n_rows", column="n")
    parts_strata = []
    for name in sorted(strata):
        if not name:
            continue
        c = tr.count(strata[name], file="audit_round1.csv", row_key=f"auto_category:{name}", column="n")
        parts_strata.append(f"{name.replace('_', ' ')} {c}")
    strata_txt = ", ".join(parts_strata)
    n_ref = tr.count(n_ref_only, file="audit_metrics.csv", row_key="n_ref_error_only", column="k")

    pm = met["precision_mandarinization"]
    pk = tr.count(pm["k"], file="audit_metrics.csv", row_key="precision_mandarinization", column="k")
    pn = tr.count(pm["n"], file="audit_metrics.csv", row_key="precision_mandarinization", column="n")
    pp = tr.pct(pm["estimate"], file="audit_metrics.csv", row_key="precision_mandarinization", column="estimate")
    pci = tr.ci_pct(pm["ci_low"], pm["ci_high"], file="audit_metrics.csv", row_key="precision_mandarinization")

    rw = met["recall_mandarinization_weighted"]
    rec = tr.pct(rw["estimate"], file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="estimate")
    rci = tr.ci_pct(rw["ci_low"], rw["ci_high"], file="audit_metrics.csv", row_key="recall_mandarinization_weighted")
    n_pop_s = tr.count(n_pop, file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="note:N_ctrl_pop")
    miss_s = tr.count(misses_ctrl, file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="note:misses_ctrl")
    den_s = tr.count(design_n, file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="note:design_n")
    b_s = tr.count(b, file="PROVENANCE.txt", row_key="B", column="B")
    seed_s = tr.count(seed, file="PROVENANCE.txt", row_key="seed", column="seed")

    po = met["precision_over_dialect"]
    ok = tr.count(po["k"], file="audit_metrics.csv", row_key="precision_over_dialect", column="k")
    on = tr.count(po["n"], file="audit_metrics.csv", row_key="precision_over_dialect", column="n")
    op = tr.pct(po["estimate"], file="audit_metrics.csv", row_key="precision_over_dialect", column="estimate")
    oci = tr.ci_pct(po["ci_low"], po["ci_high"], file="audit_metrics.csv", row_key="precision_over_dialect")
    fp_txt = ", ".join(f"`{f}`" for f in fp_forms) if fp_forms else "none listed in the guideline notes"

    setup = (
        f"We adjudicated a stratified sample of {n_s} utterances "
        f"({strata_txt}) with a single annotator who was blind to the automatic categories "
        f"and who applied multi-label error tags. Rows whose only tag is `ref_error` "
        f"(n = {n_ref}) are excluded from detector precision denominators."
    )
    mand = (
        f"The automatic mandarinization detector is **precise but narrow**. "
        f"Of {pn} automatically flagged items, {pk} carried a human mandarinization tag "
        f"(precision {pp}% , {tr.wilson_level()} {pci}). "
        f"Sampling-weighted recall is {rec}% (percentile bootstrap {tr.add(str(WILSON_PCT), file='code', row_key='bootstrap', column='percent', raw_value=WILSON_PCT)}% CI {rci}; "
        f"B = {b_s}, seed {seed_s}). "
        f"Control utterances are a sample of {den_s} from a frame of {n_pop_s} "
        f"reference-dialect items with no automatic candidate, so each control miss is "
        f"weighted by w = {n_pop_s}/{den_s} ({miss_s} misses in the sample). "
        f"When the detector fires it is usually right, but it misses most mandarinization "
        f"outside the marker lexicon; automatic counts are therefore a lower bound."
    )
    over = (
        f"The over-dialectalization detector confirmed {ok} of {on} automatic flags "
        f"(precision {op}%, {tr.wilson_level()} {oci}). "
        f"**The detector is unreliable and its automatic counts are not treated as a result.** "
        f"Dominant false-positive forms in the guideline notes include {fp_txt}."
    )
    low_para = ""
    if include_low and low_rows is not None and "control_lowconf_error_rate" in met:
        lr = met["control_lowconf_error_rate"]
        lk = tr.count(lr["k"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="k")
        ln = tr.count(lr["n"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="n")
        lp = tr.pct(lr["estimate"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="estimate")
        lci = tr.ci_pct(lr["ci_low"], lr["ci_high"], file="audit_metrics.csv", row_key="control_lowconf_error_rate")
        dist_bits = []
        for t, c in (low_dist or {}).items():
            if c:
                dist_bits.append(
                    f"{t} {tr.count(c, file='audit_round1_clean.csv', row_key=f'lowconf:{t}', column='h_is')}"
                )
        dist_txt = ", ".join(dist_bits) if dist_bits else "no error tags"
        if _lowconf_mostly_nondialect(low_dist or {}):
            mix = "Confidence therefore picks up errors, but mostly non-dialect ones (homophone and other)."
        else:
            mix = "The tag mix among these items is " + dist_txt + "."
        low_para = (
            f"Among low-confidence controls, {lk} of {ln} items had at least one human error tag "
            f"({lp}%, {tr.wilson_level()} {lci}; tags: {dist_txt}). {mix}"
        )
    ex_lines = []
    for r in examples:
        aid = r.get("audit_id") or ""
        tags = r.get("h_category") or ""
        notes = (r.get("h_notes") or "").strip()
        quote = notes[:80] if notes else ""
        ref = r.get("ref_norm") or ""
        lab = r.get("label_vote") or ""
        bit = f"{aid}: {ref} → {lab} (tags: {tags}"
        if quote:
            bit += f"; notes: “{quote}”"
        bit += ")."
        ex_lines.append(bit)
    err_para = (
        "Illustrative labelled spans (deterministic first matches; notes quoted without paraphrase) include "
        + " ".join(ex_lines)
        if ex_lines
        else "No illustrative spans were available in the joined table."
    )

    dpp = stats[("delta_pp_pass_0.9", "All")]
    lor = stats[("logit_OR_has_dialect_P0", "All")]
    dece = stats[("delta_ece_has_minus_no", "All")]
    dpp_est = float(dpp["estimate"])
    or_est = float(lor["estimate"])
    more = delta_pass_word(dpp_est)
    tr.add(more, file="stats_table.csv", row_key="delta_pp_pass_0.9", column="sign_word", raw_value=dpp_est)
    cmp_d = ">" if dpp_est > 0 else "<"
    tr.add(cmp_d, file="stats_table.csv", row_key="delta_pp_pass_0.9", column="cmp", raw_value=dpp_est)
    cmp_or = "<" if or_est < 1 else ">"
    tr.add(cmp_or, file="stats_table.csv", row_key="logit_OR_has_dialect_P0", column="cmp", raw_value=or_est)
    p_pp = float(dpp["p_holm"])
    p_or = float(lor["p_holm"])
    clause = holm_clause(p_pp, p_or)
    tr.add(clause, file="stats_table.csv", row_key="p_holm", column="clause", raw_value=f"{p_pp},{p_or}")
    dpp_s = tr.sig(dpp["estimate"], file="stats_table.csv", row_key="delta_pp_pass_0.9", column="estimate")
    or_s = tr.sig(lor["estimate"], file="stats_table.csv", row_key="logit_OR_has_dialect_P0", column="estimate")
    ece_s = tr.sig(dece["estimate"], file="stats_table.csv", row_key="delta_ece_has_minus_no", column="estimate")
    p_pp_s = tr.sig(dpp["p_holm"], file="stats_table.csv", row_key="delta_pp_pass_0.9", column="p_holm")
    p_or_s = tr.sig(lor["p_holm"], file="stats_table.csv", row_key="logit_OR_has_dialect_P0", column="p_holm")
    p_ece_s = tr.sig(dece["p_holm"], file="stats_table.csv", row_key="delta_ece_has_minus_no", column="p_holm")
    thr = tr.add(FILTER_THR, file="stats_table.csv", row_key="delta_pp_pass_0.9", column="name", raw_value=FILTER_THR)

    link = (
        f"The audit qualifies the automatic marker-retention counts; it does not change the "
        f"filter-bias conclusion. Dialect-marked utterances pass the {thr} confidence filter "
        f"**{more}** often unadjusted (Δpp = dialect − non-dialect {cmp_d} 0; "
        f"Δpp = {dpp_s}, Holm p = {p_pp_s}), the adjusted odds ratio after duration and split is "
        f"{cmp_or} 1 (OR = {or_s}, Holm p = {p_or_s}), and **{clause}**. "
        f"Dialect items are slightly worse calibrated (ΔECE = {ece_s}, Holm p = {p_ece_s}). "
        f"We do not treat this as evidence that filter bias is established."
    )
    lim = (
        "A single annotator has labelled the sample so far; intra-annotator κ on `reannotate_ids.txt` "
        "is planned after a multi-week gap. The lexicon was not retuned on this sample, "
        "so the recall figure is specific to the frozen marker list."
    )
    paras = [setup, mand, over]
    if low_para:
        paras.append(low_para)
    paras.extend([err_para, link, lim])
    return "\n\n".join(paras) + "\n"


def render_seminar(
    *,
    tr: Trace,
    met: dict[str, dict[str, str]],
    stats: dict[tuple[str, str], dict[str, str]],
    strata: dict[str, int],
    n_sample: int,
    include_low: bool,
    low_dist: dict[str, int] | None,
    n_pop: int,
    design_n: int,
) -> str:
    n_s = tr.count(n_sample, file="audit_round1.csv", row_key="n_rows", column="n")
    st = ", ".join(
        f"{k}={tr.count(v, file='audit_round1.csv', row_key=f'auto_category:{k}', column='n')}"
        for k, v in sorted(strata.items())
        if k
    )
    pm = met["precision_mandarinization"]
    pp = tr.pct(pm["estimate"], file="audit_metrics.csv", row_key="precision_mandarinization", column="estimate")
    pci = tr.ci_pct(pm["ci_low"], pm["ci_high"], file="audit_metrics.csv", row_key="precision_mandarinization")
    rw = met["recall_mandarinization_weighted"]
    rec = tr.pct(rw["estimate"], file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="estimate")
    rci = tr.ci_pct(rw["ci_low"], rw["ci_high"], file="audit_metrics.csv", row_key="recall_mandarinization_weighted")
    po = met["precision_over_dialect"]
    op = tr.pct(po["estimate"], file="audit_metrics.csv", row_key="precision_over_dialect", column="estimate")
    oci = tr.ci_pct(po["ci_low"], po["ci_high"], file="audit_metrics.csv", row_key="precision_over_dialect")
    dpp = stats[("delta_pp_pass_0.9", "All")]
    lor = stats[("logit_OR_has_dialect_P0", "All")]
    more = delta_pass_word(float(dpp["estimate"]))
    tr.add(more, file="stats_table.csv", row_key="delta_pp_pass_0.9", column="sign_word", raw_value=dpp["estimate"])
    dpp_s = tr.sig(dpp["estimate"], file="stats_table.csv", row_key="delta_pp_pass_0.9", column="estimate")
    or_s = tr.sig(lor["estimate"], file="stats_table.csv", row_key="logit_OR_has_dialect_P0", column="estimate")
    p_pp_s = tr.sig(dpp["p_holm"], file="stats_table.csv", row_key="delta_pp_pass_0.9", column="p_holm")
    p_or_s = tr.sig(lor["p_holm"], file="stats_table.csv", row_key="logit_OR_has_dialect_P0", column="p_holm")
    clause = holm_clause(float(dpp["p_holm"]), float(lor["p_holm"]))
    tr.add(clause, file="stats_table.csv", row_key="p_holm", column="clause", raw_value="seminar")
    thr = tr.add(FILTER_THR, file="stats_table.csv", row_key="delta_pp_pass_0.9", column="name", raw_value=FILTER_THR)
    pop = tr.count(n_pop, file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="note:N_ctrl_pop")
    den = tr.count(design_n, file="audit_metrics.csv", row_key="recall_mandarinization_weighted", column="note:design_n")

    b4 = "Low-confidence controls: not present in this metrics table."
    if include_low and "control_lowconf_error_rate" in met:
        lr = met["control_lowconf_error_rate"]
        lk = tr.count(lr["k"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="k")
        ln = tr.count(lr["n"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="n")
        lp = tr.pct(lr["estimate"], file="audit_metrics.csv", row_key="control_lowconf_error_rate", column="estimate")
        if _lowconf_mostly_nondialect(low_dist or {}):
            mix = "mostly homophone/other rather than dialect mapping errors"
        else:
            mix = "see tag counts in the findings"
        b4 = f"Low-confidence controls: {lk}/{ln} ({lp}%) have a human error tag; {mix}."

    lines = [
        f"- Design: {n_s} items ({st}); blind single annotator; input hashes recorded and stats_table hash unchanged.",
        f"- Mandarinization is precise but narrow: precision {pp}% {pci}; weighted recall {rec}% {rci} (w = {pop}/{den}).",
        f"- Over-dialect detector precision {op}% {oci}; **unreliable**, automatic counts are not a result.",
        f"- {b4}",
        (
            f"- Filter: dialect utterances pass the {thr} threshold **{more}** often unadjusted "
            f"(Δpp = {dpp_s}, Holm p = {p_pp_s}); OR after duration and split = {or_s} "
            f"(Holm p = {p_or_s}); {clause}; no claim of filter bias."
        ),
        "- Limitations: single annotator; κ after a multi-week gap (early December); lexicon not retuned; LLM-GER and Whisper not run.",
    ]
    return "\n".join(lines) + "\n"


def write_figure(
    path_pdf: Path,
    path_png: Path,
    *,
    mand_p: float,
    mand_p_lo: float,
    mand_p_hi: float,
    rec: float,
    rec_lo: float,
    rec_hi: float,
    over_p: float,
    over_p_lo: float,
    over_p_hi: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "wscaudit-prompt16"
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    rows = [
        ("Mandarinization precision (Wilson)", mand_p, mand_p_lo, mand_p_hi, False),
        ("Mandarinization weighted recall (bootstrap)", rec, rec_lo, rec_hi, False),
        ("Over-dialect precision (Wilson)", over_p, over_p_lo, over_p_hi, True),
    ]
    ys = [2, 1, 0]
    for y, (lab, est, lo, hi, grey) in zip(ys, rows):
        color = "#888888" if grey else "#1f4e79"
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o", color=color, capsize=3, label=lab if y == 2 or grey else None)
        if grey:
            ax.text(min(0.98, hi + 0.02), y, "unreliable detector", va="center", fontsize=8, color="#666666")
    ax.set_xlim(0, 1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Estimate")
    ax.legend(loc="lower right", fontsize=7, title="CI type")
    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, dpi=200, metadata={"CreationDate": None, "ModDate": None})
    try:
        fig.savefig(path_png, dpi=200, metadata={"CreationDate": None})
    except (TypeError, ValueError):
        fig.savefig(path_png, dpi=200)
    plt.close(fig)


def number_tokens(text: str) -> set[str]:
    masked = re.sub(r"\bA\d{3,}\b", " ", text)
    masked = re.sub(r"\b20\d{2}\b", " ", masked)
    return set(_NUM_TOKEN.findall(masked))


def run_audit_report(
    *,
    audit_dir: Path,
    stats_table: Path,
    out_dir: Path,
    repo_root: Path | None = None,
    plot: bool = True,
) -> dict[str, Path]:
    repo_root = repo_root or _REPO_ROOT
    audit_dir = Path(audit_dir)
    stats_table = Path(stats_table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "audit_metrics.csv": audit_dir / "audit_metrics.csv",
        "audit_round1_clean.csv": audit_dir / "audit_round1_clean.csv",
        "audit_round1.csv": audit_dir / "audit_round1.csv",
        "audit_merge_confusion.csv": audit_dir / "audit_merge_confusion.csv",
        "guideline_v2_notes.md": audit_dir / "guideline_v2_notes.md",
        "PROVENANCE.txt": audit_dir / "PROVENANCE.txt",
        "stats_table.csv": stats_table,
        "audit_table.tex": repo_root / "results" / "audit_table.tex",
    }
    for name, p in inputs.items():
        if name == "audit_table.tex" and not p.is_file():
            continue
        if not p.is_file():
            raise FileNotFoundError(p)

    sha_before = sha256_file(stats_table)
    sha_in = {k: sha256_file(p) for k, p in inputs.items() if p.is_file()}

    met_rows = load_csv_rows(inputs["audit_metrics.csv"])
    met = metrics_index(met_rows)
    key_rows = load_csv_rows(inputs["audit_round1.csv"])
    clean_rows = load_csv_rows(inputs["audit_round1_clean.csv"])
    joined = join_key_human(key_rows, clean_rows, expected_n=None)
    strata = strata_counts(key_rows)
    n_sample = len(key_rows)
    n_ref_only = int(float(met["n_ref_error_only"]["k"]))
    n_pop, misses_ctrl, design_n = parse_weighted_note(met["recall_mandarinization_weighted"]["note"])
    prov = _read_text(inputs["PROVENANCE.txt"])
    b, seed = parse_b_seed(prov)
    notes = _read_text(inputs["guideline_v2_notes.md"])
    fps = dominant_fp_forms(notes)
    examples = pick_examples(joined, ["mandarinization", "homophone", "other", "over_dialect"])
    include_low = "control_lowconf_error_rate" in met
    low_rows = [r for r in joined if (r.get("auto_category") or "") == "control_lowconf"] if include_low else None
    low_dist = (
        tag_distribution(low_rows, ["mandarinization", "over_dialect", "homophone", "other"]) if low_rows else None
    )
    stats_rows = load_csv_rows(stats_table)
    stats = stats_index(stats_rows)

    tr = Trace()
    findings = render_findings(
        tr=tr,
        met=met,
        stats=stats,
        strata=strata,
        n_sample=n_sample,
        n_ref_only=n_ref_only,
        n_pop=n_pop,
        misses_ctrl=misses_ctrl,
        design_n=design_n,
        b=b,
        seed=seed,
        fp_forms=fps,
        examples=examples,
        low_rows=low_rows,
        low_dist=low_dist,
        include_low=include_low,
    )
    tr_s = Trace()
    seminar = render_seminar(
        tr=tr_s,
        met=met,
        stats=stats,
        strata=strata,
        n_sample=n_sample,
        include_low=include_low,
        low_dist=low_dist,
        n_pop=n_pop,
        design_n=design_n,
    )
    sidecar = tr.records + tr_s.records

    pm = met["precision_mandarinization"]
    rw = met["recall_mandarinization_weighted"]
    po = met["precision_over_dialect"]
    pdf = fig_dir / "audit_precision_recall.pdf"
    png = fig_dir / "audit_precision_recall.png"
    if plot:
        write_figure(
            pdf,
            png,
            mand_p=float(pm["estimate"]),
            mand_p_lo=float(pm["ci_low"]),
            mand_p_hi=float(pm["ci_high"]),
            rec=float(rw["estimate"]),
            rec_lo=float(rw["ci_low"]),
            rec_hi=float(rw["ci_high"]),
            over_p=float(po["estimate"]),
            over_p_lo=float(po["ci_low"]),
            over_p_hi=float(po["ci_high"]),
        )
    else:
        pdf.write_bytes(b"%PDF-1.1\n%\n")
        png.write_bytes(b"\x89PNG\r\n\x1a\n")

    findings_path = out_dir / "audit_findings.md"
    numbers_path = out_dir / "audit_findings_numbers.json"
    seminar_path = out_dir / "seminar_2026-11-25_notes.md"
    findings_path.write_text(findings, encoding="utf-8")
    numbers_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seminar_path.write_text(seminar, encoding="utf-8")

    sha_after = sha256_file(stats_table)
    if sha_after != sha_before:
        raise RuntimeError("metrics/stats_table.csv changed; abort")

    outputs = {
        "audit_findings.md": findings_path,
        "audit_findings_numbers.json": numbers_path,
        "seminar_2026-11-25_notes.md": seminar_path,
        "audit_precision_recall.pdf": pdf,
        "audit_precision_recall.png": png,
    }
    lines = [
        f"git_commit={git_head(repo_root)}",
        *[f"input_sha256_{k}={v}" for k, v in sorted(sha_in.items())],
        *[f"output_sha256_{k}={sha256_file(p)}" for k, p in sorted(outputs.items())],
        f"stats_table_sha256_before={sha_before}",
        f"stats_table_sha256_after={sha_after}",
        "unweighted recall row present but not reported",
        "",
    ]
    report_prov = out_dir / "AUDIT_REPORT_PROVENANCE.txt"
    text = "\n".join(lines)
    report_prov.write_text(text, encoding="utf-8")
    drive_prov = audit_dir / "PROVENANCE.txt"
    if drive_prov.is_file():
        with drive_prov.open("a", encoding="utf-8") as f:
            f.write("\n# audit-report\n")
            f.write(text)

    print(f"wrote {findings_path}")
    print(f"wrote {seminar_path}")
    print(f"wrote {pdf}")
    return {"findings": findings_path, "seminar": seminar_path, "provenance": report_prov}


def main_paths_from_env(
    *,
    audit_dir: Path | None,
    stats_table: Path | None,
    out_dir: Path | None,
    out_root: Path,
    repo_root: Path,
) -> tuple[Path, Path, Path]:
    ad = audit_dir or (out_root / "audit")
    st = stats_table or (out_root / "metrics" / "stats_table.csv")
    od = out_dir or (repo_root / "results")
    return ad, st, od
