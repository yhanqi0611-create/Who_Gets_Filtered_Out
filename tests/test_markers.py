from __future__ import annotations

import json
from pathlib import Path

from wscaudit.metrics.markers import (
    SUMMARY_KEYS,
    find_markers,
    load_markers,
    run_markers,
    score_utt,
    _dialect_mandarin_map,
    _scan_entries,
)

TOY = [
    {"dialect": "晓得", "mandarin": "知道", "note": ""},
    {"dialect": "啥子", "mandarin": "什么", "note": ""},
    {"dialect": "瓜娃子", "mandarin": "傻瓜", "note": ""},
    {"dialect": "瓜", "mandarin": "傻", "note": ""},
    {"dialect": "噻", "mandarin": "", "note": "particle"},
    {"dialect": "嘿", "mandarin": "很", "note": ""},
]


def test_find_晓得_啥子() -> None:
    hits = find_markers("我晓得啥子", pairs=TOY)
    assert [h["surface"] for h in hits if h["kind"] == "dialect"] == ["晓得", "啥子"]


def test_xigua_does_not_match_gua() -> None:
    hits = find_markers("西瓜", pairs=TOY)
    assert hits == []


def test_longest_first_guawazi() -> None:
    hits = find_markers("瓜娃子", pairs=TOY)
    assert [h["surface"] for h in hits] == ["瓜娃子"]
    assert hits[0]["kind"] == "dialect"


def test_particle_sai() -> None:
    hits = find_markers("好巴适噻", pairs=TOY + [{"dialect": "巴适", "mandarin": "舒服", "note": ""}])
    surfaces = [h["surface"] for h in hits if h["kind"] == "dialect"]
    assert "噻" in surfaces


def test_hei_prefix_ok() -> None:
    hits = find_markers("嘿安逸", pairs=TOY + [{"dialect": "安逸", "mandarin": "舒服", "note": ""}])
    assert [h["surface"] for h in hits if h["kind"] == "dialect"] == ["嘿", "安逸"]


def test_real_lexicon_loads() -> None:
    rows = load_markers()
    assert len(rows) >= 30
    dialects = {r["dialect"] for r in rows}
    assert "啥子" in dialects and "噻" in dialects and "瓜娃子" in dialects
    assert "瓜" not in dialects


def test_run_markers_summary_schema(tmp_path: Path) -> None:
    lex = tmp_path / "markers.tsv"
    lex.write_text(
        "dialect\tmandarin\tnote\n"
        "晓得\t知道\t\n"
        "啥子\t什么\t\n"
        "噻\t\tparticle\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    merged = out / "merged"
    merged.mkdir(parents=True)
    man = out / "manifest.jsonl"
    rows = [
        ("u1", "Hard", "我晓得啥子", "我晓得啥子", 1.0),
        ("u2", "Easy", "啥子", "什么", 0.5),
        ("u3", "Easy", "你好", "你好", 0.95),
    ]
    with man.open("w", encoding="utf-8") as f:
        for uid, spl, ref, _h, _c in rows:
            f.write(json.dumps({"utt_id": uid, "split": spl, "wav": "x", "ref": ref, "dur": 1}) + "\n")
    with (merged / "vote_P0.jsonl").open("w", encoding="utf-8") as f:
        for uid, spl, _r, hyp, conf in rows:
            f.write(json.dumps({"utt_id": uid, "split": spl, "label": hyp, "conf": conf}) + "\n")
    summary = run_markers(man, merged / "vote_P0.jsonl", markers_path=lex, metrics_dir=out / "metrics")
    assert SUMMARY_KEYS <= set(summary)
    assert summary["n"] == 3
    assert summary["lexicon_n"] == 3
    assert summary["has_dialect_ref"]["All"]["n_utt"] == 2
    assert summary["mandarinization_n"]["All"] == 1
    per = [json.loads(x) for x in (out / "metrics" / "markers_per_utt.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(per) == 3
    by_id = {r["utt_id"]: r for r in per}
    assert by_id["u2"]["mandarinization_n"] == 1
    assert by_id["u1"]["retained_token_frac"] == 1.0
    assert by_id["u3"]["has_dialect_ref"] is False


def test_score_over_dialect() -> None:
    scan = _scan_entries(TOY)
    d2m = _dialect_mandarin_map(TOY)
    out = score_utt("什么", "啥子", TOY, scan, d2m)
    assert out["over_dialect_n"] == 1
    assert out["has_dialect_ref"] is False
