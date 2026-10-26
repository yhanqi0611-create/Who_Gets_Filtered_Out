from __future__ import annotations

from pathlib import Path

from wscaudit.normalize import (
    DEFAULT_NORM,
    light_normalize,
    load_equivalences,
    normalize,
    report_ref_diffs,
)


def _cfg(**overrides):
    cfg = dict(DEFAULT_NORM)
    cfg.update(overrides)
    return cfg


def test_traditional_to_simplified() -> None:
    assert normalize("這", _cfg()) == "这"
    # OpenCC t2s also maps some already-Simplified variants (e.g. 乾 → 干).
    assert normalize("乾德", _cfg()) == "干德"


def test_punctuation_stripped() -> None:
    assert normalize("好巴适。", _cfg()) == "好巴适"
    assert normalize("好巴适·边", _cfg()) == "好巴适边"


def test_sensevoice_tags_removed() -> None:
    raw = "<|zh|><|NEUTRAL|><|Speech|>诶呀好巴适<|woitn|>"
    assert normalize(raw, _cfg()) == "诶呀好巴适"


def test_arabic_digits_to_cn_logged() -> None:
    log: list[tuple[str, str]] = []
    assert normalize("3个", _cfg(), digit_log=log) == "三个"
    assert log == [("3", "三")]


def test_digits_can_be_disabled() -> None:
    assert normalize("3个", _cfg(arabic_digits_to_cn=False)) == "3个"


def test_whitespace_removed() -> None:
    assert normalize("好 巴适", _cfg()) == "好巴适"
    assert normalize("hello 世界", _cfg()) == "hello世界"


def test_latin_lowercased() -> None:
    assert normalize("OK巴适", _cfg()) == "ok巴适"


def test_equivalences_optional(tmp_path: Path) -> None:
    tsv = tmp_path / "eq.tsv"
    tsv.write_text("variant\tcanonical\n噻\t撒\n", encoding="utf-8")
    pairs = load_equivalences(tsv)
    off = normalize("好噻", _cfg(apply_equivalences=False), equivalences=pairs)
    on = normalize("好噻", _cfg(apply_equivalences=True), equivalences=pairs)
    assert off == "好噻"
    assert on == "好撒"


def test_light_normalize_keeps_middle_dot() -> None:
    assert "·" in light_normalize("这·边还有一块嗯")
    assert "·" not in normalize("这·边还有一块嗯", _cfg())


def test_report_ref_diffs_classifies_middle_dot() -> None:
    refs = [
        ("a", "诶呀好巴适"),
        ("b", "hello 世界"),
        ("c", "这·边还有一块嗯"),
    ]
    summary = report_ref_diffs(refs, _cfg())
    assert summary["n"] == 3
    assert summary["whitespace_or_latin_case_only"] == 1
    assert summary["other"] == 1
    assert summary["other_examples"][0]["utt_id"] == "c"
