from __future__ import annotations

import csv
from pathlib import Path

from wscaudit.audit import BLIND_DROP, FULL_COLS, merge_audit, select_audit_utts, write_csv


def _marks() -> list[dict]:
    rows = []
    # 5 mandarinization, 4 over, 1 both
    for i in range(5):
        rows.append(
            {
                "utt_id": f"m{i}",
                "split": "Easy",
                "system": "vote_P0",
                "conf": 0.9,
                "has_dialect_ref": True,
                "mandarinization_n": 1,
                "over_dialect_n": 0,
                "ref_dialect": ["啥子"],
                "hyp_dialect": [],
            }
        )
    for i in range(4):
        rows.append(
            {
                "utt_id": f"o{i}",
                "split": "Hard",
                "system": "vote_P0",
                "conf": 0.85,
                "has_dialect_ref": False,
                "mandarinization_n": 0,
                "over_dialect_n": 1,
                "ref_dialect": [],
                "hyp_dialect": ["啥子"],
            }
        )
    rows.append(
        {
            "utt_id": "b0",
            "split": "Easy",
            "system": "vote_P0",
            "conf": 0.8,
            "has_dialect_ref": True,
            "mandarinization_n": 1,
            "over_dialect_n": 1,
            "ref_dialect": ["晓得"],
            "hyp_dialect": ["啥子"],
        }
    )
    for i in range(50):
        rows.append(
            {
                "utt_id": f"d{i}",
                "split": "Easy",
                "system": "vote_P0",
                "conf": 0.95,
                "has_dialect_ref": True,
                "mandarinization_n": 0,
                "over_dialect_n": 0,
                "ref_dialect": ["巴适"],
                "hyp_dialect": ["巴适"],
            }
        )
    for i in range(30):
        rows.append(
            {
                "utt_id": f"l{i}",
                "split": "Hard",
                "system": "vote_P0",
                "conf": 0.4 if i < 15 else 0.95,
                "has_dialect_ref": False,
                "mandarinization_n": 0,
                "over_dialect_n": 0,
                "ref_dialect": [],
                "hyp_dialect": [],
            }
        )
    return rows


def _cer(marks: list[dict]) -> dict[str, dict]:
    out = {}
    for r in marks:
        uid = r["utt_id"]
        cer = 0.6 if uid.startswith("l") and r["conf"] >= 0.9 else 0.1
        out[uid] = {"utt_id": uid, "cer": cer, "system": "vote_P0"}
    return out


def test_select_deterministic() -> None:
    marks = _marks()
    cer = _cer(marks)
    a, ca = select_audit_utts(marks, cer, seed=20261103, n_dialect=40, n_low=20)
    b, cb = select_audit_utts(marks, cer, seed=20261103, n_dialect=40, n_low=20)
    assert [r["utt_id"] for r in a] == [r["utt_id"] for r in b]
    assert ca == cb
    assert ca["candidates_total"] == 10
    assert ca["control_dialect"] == 40
    assert ca["control_lowconf"] == 20
    assert ca["mandarinization"] == 5
    assert ca["over_dialect"] == 4
    assert ca["both"] == 1
    assert ca["total"] == 70


def test_blind_columns(tmp_path: Path) -> None:
    rows = [
        {c: "x" for c in FULL_COLS}
        | {"audit_id": "A001", "auto_category": "mandarinization", "conf": 0.9, "cer_vote": 0.2}
    ]
    write_csv(tmp_path / "full.csv", rows, FULL_COLS)
    blind_cols = [c for c in FULL_COLS if c not in BLIND_DROP]
    write_csv(tmp_path / "blind.csv", rows, blind_cols)
    with (tmp_path / "blind.csv").open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert "auto_category" not in header
    assert "conf" not in header
    assert "cer_vote" not in header
    assert "audit_id" in header


def test_audit_merge_confusion(tmp_path: Path) -> None:
    key_rows = [
        {"audit_id": "A001", "auto_category": "mandarinization", "h_category": ""},
        {"audit_id": "A002", "auto_category": "over_dialect", "h_category": ""},
        {"audit_id": "A003", "auto_category": "control_dialect", "h_category": ""},
    ]
    human_rows = [
        {"audit_id": "A001", "h_category": "mandarinization"},
        {"audit_id": "A002", "h_category": "no_error"},
        {"audit_id": "A003", "h_category": "no_error"},
    ]
    write_csv(tmp_path / "key.csv", key_rows, ["audit_id", "auto_category", "h_category"])
    write_csv(tmp_path / "human.csv", human_rows, ["audit_id", "h_category"])
    out = merge_audit(tmp_path / "key.csv", tmp_path / "human.csv")
    assert out["n"] == 3
    assert out["confusion"]["mandarinization|mandarinization"] == 1
    assert out["confusion"]["over_dialect|no_error"] == 1
    assert out["confusion"]["control_dialect|no_error"] == 1
