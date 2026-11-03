from __future__ import annotations

import json
from pathlib import Path

import pytest

from wscaudit.metrics.calibration import (
    CalibrationJoinError,
    acc_from_cer,
    ece_and_bins,
    join_rows,
    run_calibration,
    spearman_pair,
)


def test_perfect_conf_equals_acc_ece_zero() -> None:
    rows = [
        {"conf": 0.2, "acc": 0.2},
        {"conf": 0.7, "acc": 0.7},
        {"conf": 0.95, "acc": 0.95},
        {"conf": 0.95, "acc": 0.95},
    ]
    ece, details = ece_and_bins(rows)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert all(d["gap"] in (0.0, None) or d["gap"] == pytest.approx(0.0) for d in details)


def test_cer_clip_for_acc() -> None:
    assert acc_from_cer(0.25) == 0.75
    assert acc_from_cer(1.5) == 0.0


def test_three_utt_spearman_sign() -> None:
    # conf ↑ as cer ↓ → ρ(conf, cer) = −1
    confs = [0.2, 0.5, 0.9]
    cers = [0.8, 0.4, 0.1]
    accs = [acc_from_cer(c) for c in cers]
    cer_sp = spearman_pair(confs, cers)
    acc_sp = spearman_pair(confs, accs)
    assert cer_sp["rho"] == pytest.approx(-1.0)
    assert acc_sp["rho"] is not None and acc_sp["rho"] > 0


def _write_triple(tmp_path: Path, rows: list[tuple[str, str, float, float, bool]]) -> tuple[Path, Path, Path]:
    vote = tmp_path / "vote.jsonl"
    cer = tmp_path / "cer.jsonl"
    mark = tmp_path / "mark.jsonl"
    with vote.open("w", encoding="utf-8") as fv, cer.open("w", encoding="utf-8") as fc, mark.open("w", encoding="utf-8") as fm:
        for uid, spl, conf, cer_v, has in rows:
            fv.write(json.dumps({"utt_id": uid, "split": spl, "label": "x", "conf": conf}) + "\n")
            fc.write(json.dumps({"utt_id": uid, "split": spl, "system": "vote_P0", "cer": cer_v, "ref_len": 4, "distance": 1}) + "\n")
            fm.write(json.dumps({"utt_id": uid, "split": spl, "has_dialect_ref": has}) + "\n")
    return vote, cer, mark


def test_join_missing_key_raises(tmp_path: Path) -> None:
    vote, cer, mark = _write_triple(
        tmp_path,
        [
            ("u1", "Hard", 0.9, 0.1, True),
            ("u2", "Easy", 0.5, 0.4, False),
        ],
    )
    extra = json.loads(cer.read_text(encoding="utf-8").splitlines()[0])
    extra["utt_id"] = "u3"
    with cer.open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra) + "\n")
    with pytest.raises(CalibrationJoinError, match="join failed"):
        join_rows(vote, cer, mark, expected_n=2)


def test_run_calibration_toy(tmp_path: Path) -> None:
    vote, cer, mark = _write_triple(
        tmp_path,
        [
            ("u1", "Hard", 0.2, 0.8, True),
            ("u2", "Easy", 0.5, 0.4, False),
            ("u3", "Easy", 0.9, 0.1, True),
        ],
    )
    metrics = tmp_path / "metrics"
    summary = run_calibration(vote, cer, mark, metrics, expected_n=3, plot=False)
    assert summary["n"] == 3
    assert summary["groups"]["All"]["spearman_conf_cer"]["rho"] < 0
    assert (metrics / "calibration_summary.json").is_file()
    assert (metrics / "calibration_bins.csv").is_file()
    assert summary["groups"]["All_has_dialect_ref"]["n"] == 2
