from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from wscaudit.audit_adjudicate import sha256_file, write_csv
from wscaudit.audit_report import (
    Trace,
    delta_pass_word,
    dominant_fp_forms,
    holm_clause,
    holm_significant,
    number_tokens,
    pick_examples,
    run_audit_report,
)


def test_formatter_pct_and_ci() -> None:
    tr = Trace()
    assert tr.pct(0.7188, file="f", row_key="r", column="estimate") == "71.9"
    tr2 = Trace()
    assert tr2.ci_pct(0.546, 0.844, file="f", row_key="r") == "[54.6, 84.4]"
    assert tr.count(32, file="f", row_key="r", column="n") == "32"
    assert tr.sig(0.0125, file="f", row_key="r", column="e") == "0.0125"


def test_number_tokens_skip_ids_and_years() -> None:
    text = "A001 in 2026 had 23 of 32 (71.9%)."
    toks = number_tokens(text)
    assert "23" in toks and "32" in toks and "71.9" in toks
    assert "001" not in toks
    assert "2026" not in toks


def test_sign_wording() -> None:
    assert delta_pass_word(3.36) == "more"
    assert delta_pass_word(-1.0) == "less"
    assert holm_significant(0.0534) is False
    assert holm_significant(0.01) is True
    assert holm_clause(0.0534, 0.0534) == "neither is significant after Holm"


def test_pick_examples_prefers_notes() -> None:
    rows = [
        {"audit_id": "A002", "h_is_homophone": "1", "h_notes": ""},
        {"audit_id": "A001", "h_is_homophone": "1", "h_notes": "span"},
        {"audit_id": "A003", "h_is_other": "1", "h_notes": "x"},
    ]
    got = pick_examples(rows, ["mandarinization", "homophone", "other", "over_dialect"])
    ids = [r["audit_id"] for r in got]
    assert ids[0] == "A001"
    assert "A003" in ids


def test_dominant_fp_forms() -> None:
    notes = """## Auto over_dialect false positives

- `嘞`: n=9; human tags: 'other'×5
- `噻`: n=3; human tags: 'homophone'×3

## Next
"""
    assert dominant_fp_forms(notes)[0] == "嘞"


def _write_fixture(root: Path, *, with_low: bool = True) -> tuple[Path, Path, Path]:
    audit = root / "audit"
    metrics = root / "metrics"
    results = root / "results"
    audit.mkdir()
    metrics.mkdir()
    results.mkdir()
    key = [
        {"audit_id": "A001", "auto_category": "mandarinization", "ref_norm": "一哈", "label_vote": "一下"},
        {"audit_id": "A002", "auto_category": "over_dialect", "ref_norm": "呢", "label_vote": "嘞"},
        {"audit_id": "A003", "auto_category": "control_dialect", "ref_norm": "啥子", "label_vote": "什么"},
        {"audit_id": "A004", "auto_category": "control_lowconf", "ref_norm": "去", "label_vote": "绿马"},
    ]
    clean = [
        {
            "audit_id": "A001",
            "h_category": "mandarinization",
            "h_is_mandarinization": "1",
            "h_is_over_dialect": "0",
            "h_is_homophone": "0",
            "h_is_other": "0",
            "h_any_error": "1",
            "h_notes": "一哈→一下",
            "ref_norm": "一哈",
            "label_vote": "一下",
        },
        {
            "audit_id": "A002",
            "h_category": "other",
            "h_is_mandarinization": "0",
            "h_is_over_dialect": "0",
            "h_is_homophone": "0",
            "h_is_other": "1",
            "h_any_error": "1",
            "h_notes": "",
            "ref_norm": "呢",
            "label_vote": "嘞",
        },
        {
            "audit_id": "A003",
            "h_category": "mandarinization|homophone",
            "h_is_mandarinization": "1",
            "h_is_over_dialect": "0",
            "h_is_homophone": "1",
            "h_is_other": "0",
            "h_any_error": "1",
            "h_notes": "",
            "ref_norm": "啥子",
            "label_vote": "什么",
        },
        {
            "audit_id": "A004",
            "h_category": "homophone",
            "h_is_mandarinization": "0",
            "h_is_over_dialect": "0",
            "h_is_homophone": "1",
            "h_is_other": "0",
            "h_any_error": "1",
            "h_notes": "立马",
            "ref_norm": "去",
            "label_vote": "绿马",
        },
    ]
    if not with_low:
        key = [r for r in key if r["auto_category"] != "control_lowconf"]
        clean = [r for r in clean if r["audit_id"] != "A004"]
    write_csv(audit / "audit_round1.csv", key, list(key[0].keys()))
    write_csv(audit / "audit_round1_clean.csv", clean, list(clean[0].keys()))
    met = [
        {
            "metric": "precision_mandarinization",
            "k": "1",
            "n": "1",
            "estimate": "1.0",
            "ci_low": "0.2",
            "ci_high": "1.0",
            "note": "Wilson",
        },
        {
            "metric": "precision_over_dialect",
            "k": "0",
            "n": "1",
            "estimate": "0.0",
            "ci_low": "0.0",
            "ci_high": "0.8",
            "note": "Wilson",
        },
        {
            "metric": "recall_mandarinization_weighted",
            "k": "1",
            "n": "100",
            "estimate": "0.05",
            "ci_low": "0.02",
            "ci_high": "0.08",
            "note": "TP_cand/(TP_cand+w*misses_ctrl); w=N_ctrl_pop/40=2.5; N_ctrl_pop=100; misses_ctrl=1",
        },
        {
            "metric": "recall_mandarinization_unweighted",
            "k": "1",
            "n": "2",
            "estimate": "0.5",
            "ci_low": "",
            "ci_high": "",
            "note": "UNWEIGHTED – do not report as recall 1/2",
        },
        {"metric": "n_ref_error_only", "k": "0", "n": str(len(key)), "estimate": "", "ci_low": "", "ci_high": "", "note": ""},
        {"metric": "human_over_dialect_n", "k": "0", "n": str(len(key)), "estimate": "", "ci_low": "", "ci_high": "", "note": ""},
    ]
    if with_low:
        met.append(
            {
                "metric": "control_lowconf_error_rate",
                "k": "1",
                "n": "1",
                "estimate": "1.0",
                "ci_low": "0.2",
                "ci_high": "1.0",
                "note": "h_any_error Wilson 95%",
            }
        )
    write_csv(audit / "audit_metrics.csv", met, ["metric", "k", "n", "estimate", "ci_low", "ci_high", "note"])
    write_csv(
        audit / "audit_merge_confusion.csv",
        [{"auto_category": "mandarinization", "mandarinization": "1"}],
        ["auto_category", "mandarinization"],
    )
    (audit / "guideline_v2_notes.md").write_text(
        "## Auto over_dialect false positives\n\n- `嘞`: n=4; human tags: 'other'×4\n\n## Open\n",
        encoding="utf-8",
    )
    (audit / "PROVENANCE.txt").write_text("B=10000\nseed=20261103\n", encoding="utf-8")
    stats = [
        {
            "name": "delta_pp_pass_0.9",
            "split": "All",
            "estimate": "3.36",
            "ci_lo": "0.77",
            "ci_hi": "5.93",
            "p_raw": "0.027",
            "p_holm": "0.0534",
            "family": "primary",
        },
        {
            "name": "logit_OR_has_dialect_P0",
            "split": "All",
            "estimate": "0.843",
            "ci_lo": "0.72",
            "ci_hi": "0.98",
            "p_raw": "0.028",
            "p_holm": "0.0534",
            "family": "primary",
        },
        {
            "name": "delta_ece_has_minus_no",
            "split": "All",
            "estimate": "0.0125",
            "ci_lo": "0.003",
            "ci_hi": "0.022",
            "p_raw": "0.009",
            "p_holm": "0.0258",
            "family": "primary",
        },
    ]
    write_csv(
        metrics / "stats_table.csv",
        stats,
        ["name", "split", "estimate", "ci_lo", "ci_hi", "p_raw", "p_holm", "family"],
    )
    (results / "audit_table.tex").write_text("% fixture\n", encoding="utf-8")
    return audit, metrics / "stats_table.csv", results


def test_omits_lowconf_paragraph(tmp_path: Path) -> None:
    audit, stats, out = _write_fixture(tmp_path, with_low=False)
    run_audit_report(audit_dir=audit, stats_table=stats, out_dir=out, repo_root=tmp_path, plot=False)
    text = (out / "audit_findings.md").read_text(encoding="utf-8")
    assert "low-confidence" not in text.lower()
    assert "precise but narrow" in text
    assert "unreliable" in text
    assert "23/45" not in text
    assert "naive_unweighted" not in text
    assert "bias is proven" not in text
    assert "proves" not in text


def test_run_twice_identical(tmp_path: Path) -> None:
    audit, stats, out = _write_fixture(tmp_path, with_low=True)
    run_audit_report(audit_dir=audit, stats_table=stats, out_dir=out, repo_root=tmp_path, plot=False)
    a = (out / "audit_findings.md").read_bytes()
    ja = (out / "audit_findings_numbers.json").read_bytes()
    sa = (out / "seminar_2026-11-25_notes.md").read_bytes()
    run_audit_report(audit_dir=audit, stats_table=stats, out_dir=out, repo_root=tmp_path, plot=False)
    assert (out / "audit_findings.md").read_bytes() == a
    assert (out / "audit_findings_numbers.json").read_bytes() == ja
    assert (out / "seminar_2026-11-25_notes.md").read_bytes() == sa
    notes = (out / "seminar_2026-11-25_notes.md").read_text(encoding="utf-8").strip().splitlines()
    assert len(notes) == 6
    assert notes[-1].startswith("- Limitations")
    md = (out / "audit_findings.md").read_text(encoding="utf-8")
    side = json.loads((out / "audit_findings_numbers.json").read_text(encoding="utf-8"))
    formatted = {e["formatted"] for e in side}
    for tok in number_tokens(md):
        assert tok in formatted, tok
    assert sha256_file(stats) == sha256_file(stats)


def _real_audit_metrics() -> Path | None:
    env = os.environ.get("WSC_OUT")
    if env:
        p = Path(env) / "audit" / "audit_metrics.csv"
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(_real_audit_metrics() is None, reason="audit_metrics.csv not found (Drive/WSC_OUT not mounted)")
def test_real_data_consistency() -> None:
    root = Path(__file__).resolve().parents[1]
    wsc_out = Path(os.environ["WSC_OUT"])
    audit = wsc_out / "audit"
    stats = wsc_out / "metrics" / "stats_table.csv"
    out = root / "results"
    sha_m_before = sha256_file(audit / "audit_metrics.csv")
    sha_c_before = sha256_file(audit / "audit_round1_clean.csv")
    run_audit_report(audit_dir=audit, stats_table=stats, out_dir=out, repo_root=root)
    md = (out / "audit_findings.md").read_text(encoding="utf-8")
    sem = (out / "seminar_2026-11-25_notes.md").read_text(encoding="utf-8")
    side = json.loads((out / "audit_findings_numbers.json").read_text(encoding="utf-8"))
    formatted = {e["formatted"] for e in side}
    for tok in number_tokens(md) | number_tokens(sem):
        assert tok in formatted, tok
    for e in side:
        src = Path(e["file"])
        # re-format identity is implied by Trace; skip non-files
        assert e["formatted"]
    prov = (out / "AUDIT_REPORT_PROVENANCE.txt").read_text(encoding="utf-8")
    assert f"stats_table_sha256_after={sha256_file(stats)}" in prov
    assert sha256_file(audit / "audit_metrics.csv") == sha_m_before
    assert sha256_file(audit / "audit_round1_clean.csv") == sha_c_before
    git = subprocess.run(
        ["git", "show", "e23509f:results/audit_table.tex"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    assert (root / "results" / "audit_table.tex").read_bytes() == git.stdout
    assert "23/45" not in md and "naive_unweighted" not in md
    assert "unreliable" in md and "precise but narrow" in md
    assert "bias is proven" not in md and "proves" not in md.lower()
