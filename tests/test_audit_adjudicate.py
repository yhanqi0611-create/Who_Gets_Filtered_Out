from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from wscaudit.audit import merge_audit, write_csv
from wscaudit.audit_adjudicate import (
    AuditTagError,
    bootstrap_weighted_recall,
    canonical_category,
    canonical_tags,
    cleanup_human,
    compute_metrics,
    join_key_human,
    run_adjudication,
    weighted_recall,
    wilson_ci,
)


def test_canonical_tag_order() -> None:
    a = canonical_category("other|homophone", audit_id="A1")
    b = canonical_category("homophone | other", audit_id="A1")
    assert a == b == "homophone|other"
    assert canonical_tags("other|homophone") == ["homophone", "other"]


def test_unknown_tag_raises() -> None:
    with pytest.raises(AuditTagError, match="A099"):
        canonical_tags("typo", audit_id="A099")


def test_weighted_recall_toy() -> None:
    tp = np.array([1, 1, 0, 0])
    miss = np.array([1, 1, 0, 0])
    n_pop, n_design = 40, 4
    hat = weighted_recall(tp, miss, n_pop, n_design)
    assert hat == pytest.approx(2 / (2 + 10 * 2))
    rec, ci = bootstrap_weighted_recall(tp, miss, n_pop, B=200, seed=20261103, n_ctrl_design=n_design)
    assert rec == pytest.approx(hat)
    assert ci[0] <= hat <= ci[1]


def test_wilson_matches_statsmodels() -> None:
    for k, n in ((23, 32), (3, 34)):
        lo, hi = wilson_ci(k, n)
        exp = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert lo == pytest.approx(float(exp[0]))
        assert hi == pytest.approx(float(exp[1]))


def test_cleanup_and_join(tmp_path: Path) -> None:
    human = [
        {
            "audit_id": "A001",
            "h_category": "other | homophone",
            "h_variety": "Sichuanese",
            "h_ref_correct": "Y",
        },
        {
            "audit_id": "A002",
            "h_category": "ref_error",
            "h_variety": "",
            "h_ref_correct": "n",
        },
    ]
    clean, log, stats = cleanup_human(human)
    assert clean[0]["h_category"] == "homophone|other"
    assert clean[0]["h_is_homophone"] == "1"
    assert clean[0]["h_any_error"] == "1"
    assert clean[1]["h_is_ref_error"] == "1"
    assert clean[1]["h_any_error"] == "0"
    assert clean[1]["h_variety"] == "missing"
    assert stats["n_variety_missing"] == 1
    assert any(x["column"] == "h_category" for x in log)
    key = [
        {"audit_id": "A001", "auto_category": "mandarinization"},
        {"audit_id": "A002", "auto_category": "control_dialect"},
    ]
    joined = join_key_human(key, clean, expected_n=2)
    assert len(joined) == 2
    m = compute_metrics(joined, n_ctrl_pop=100, B=50, seed=20261103)
    assert m["n_ref_error_only"] == 1
    assert m["precision_mandarinization"]["n"] == 1
    assert m["precision_mandarinization"]["k"] == 0


def test_run_adjudication_writes(tmp_path: Path) -> None:
    key_rows = [
        {
            "audit_id": "A001",
            "utt_id": "u1",
            "auto_category": "mandarinization",
            "ref_norm": "啥子",
            "label_vote": "什么",
            "ref_dialect_hits": "啥子",
            "label_dialect_hits": "",
        },
        {
            "audit_id": "A002",
            "utt_id": "u2",
            "auto_category": "over_dialect",
            "ref_norm": "什么",
            "label_vote": "啥子",
            "ref_dialect_hits": "",
            "label_dialect_hits": "啥子",
        },
        {
            "audit_id": "A003",
            "utt_id": "u3",
            "auto_category": "control_dialect",
            "ref_norm": "莫得怕",
            "label_vote": "我的怕",
            "ref_dialect_hits": "",
            "label_dialect_hits": "",
        },
        {
            "audit_id": "A004",
            "utt_id": "u4",
            "auto_category": "control_lowconf",
            "ref_norm": "你好",
            "label_vote": "你好",
            "ref_dialect_hits": "",
            "label_dialect_hits": "",
        },
    ]
    human_rows = [
        {"audit_id": "A001", "h_category": "mandarinization", "h_variety": "sichuanese", "h_ref_correct": "y"},
        {"audit_id": "A002", "h_category": "no_error", "h_variety": "sichuanese", "h_ref_correct": "y"},
        {
            "audit_id": "A003",
            "h_category": "homophone | mandarinization",
            "h_variety": "sichuanese",
            "h_ref_correct": "y",
        },
        {"audit_id": "A004", "h_category": "other", "h_variety": "accented_mandarin", "h_ref_correct": "y"},
    ]
    key_p = tmp_path / "key.csv"
    human_p = tmp_path / "human.csv"
    write_csv(key_p, key_rows, list(key_rows[0].keys()))
    write_csv(human_p, human_rows, list(human_rows[0].keys()))
    marks = tmp_path / "markers_per_utt.jsonl"
    recs = [
        {"utt_id": "u1", "system": "vote_P0", "has_dialect_ref": True, "mandarinization_n": 1, "over_dialect_n": 0},
        {"utt_id": "u2", "system": "vote_P0", "has_dialect_ref": False, "mandarinization_n": 0, "over_dialect_n": 1},
        {"utt_id": "u3", "system": "vote_P0", "has_dialect_ref": True, "mandarinization_n": 0, "over_dialect_n": 0},
    ]
    for i in range(10):
        recs.append(
            {
                "utt_id": f"d{i}",
                "system": "vote_P0",
                "has_dialect_ref": True,
                "mandarinization_n": 0,
                "over_dialect_n": 0,
            }
        )
    marks.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    stats = tmp_path / "stats_table.csv"
    stats.write_text("x,primary\n1,primary\n", encoding="utf-8")
    out = tmp_path / "audit"
    tex = tmp_path / "results" / "audit_table.tex"
    run_adjudication(
        key_path=key_p,
        human_path=human_p,
        write_dir=out,
        markers_jsonl=marks,
        tex_path=tex,
        expected_n=4,
        B=50,
        stats_table=stats,
    )
    assert (out / "audit_round1_clean.csv").is_file()
    assert (out / "audit_round1_blind_filled.csv").is_file()
    assert (out / "audit_metrics.csv").is_file()
    assert "Mandarinization" in tex.read_text(encoding="utf-8")
    assert "UNWEIGHTED" in (out / "audit_metrics.csv").read_text(encoding="utf-8")
    clean = (out / "audit_round1_clean.csv").read_text(encoding="utf-8")
    assert "mandarinization|homophone" in clean
    assert stats.read_text(encoding="utf-8") == "x,primary\n1,primary\n"
    old = merge_audit(key_p, human_p)
    assert old["n"] == 4
