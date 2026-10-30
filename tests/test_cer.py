from __future__ import annotations

import json
from pathlib import Path

from wscaudit.metrics.cer import micro_cer, run_cer, utt_distance_cer


def test_identical_cer_zero() -> None:
    d, cer, empty = utt_distance_cer("巴适得很", "巴适得很")
    assert d == 0
    assert cer == 0.0
    assert empty is False


def test_empty_hyp_nonempty_ref() -> None:
    d, cer, empty = utt_distance_cer(None, "瓜娃子")
    assert empty is True
    assert d == 3
    assert cer == 1.0
    d2, cer2, empty2 = utt_distance_cer("", "瓜娃子")
    assert empty2 is True and d2 == 3 and cer2 == 1.0


def test_empty_hyp_empty_ref() -> None:
    d, cer, empty = utt_distance_cer(None, "")
    assert empty is True
    assert d == 0
    assert cer == 0.0


def test_one_sub_in_four() -> None:
    d, cer, empty = utt_distance_cer("ABCD", "ABXD")
    assert empty is False
    assert d == 1
    assert cer == 0.25


def test_micro_not_macro() -> None:
    # 0/2 and 2/2 → micro 2/4 = 0.5 (formula in the prompt)
    assert micro_cer(0 + 2, 2 + 2) == 0.5
    # distinct from a 0/4 + 2/2 macro-style mean
    assert micro_cer(0 + 2, 4 + 2) == 2 / 6


def test_run_cer_tables(tmp_path: Path) -> None:
    out = tmp_path / "out"
    hyps = out / "hyps"
    merged = out / "merged"
    hyps.mkdir(parents=True)
    merged.mkdir()
    man = out / "manifest.jsonl"
    rows = [
        ("u1", "Hard", "巴适"),
        ("u2", "Hard", "瓜娃子"),
        ("u3", "Easy", "要得"),
        ("u4", "Easy", "巴适"),
    ]
    with man.open("w", encoding="utf-8") as f:
        for uid, spl, ref in rows:
            f.write(json.dumps({"utt_id": uid, "split": spl, "wav": "x", "ref": ref, "dur": 1}) + "\n")
    # sensevoice: all match
    with (hyps / "sensevoice.jsonl").open("w", encoding="utf-8") as f:
        for uid, _s, ref in rows:
            f.write(json.dumps({"utt_id": uid, "norm": ref, "raw": ref}) + "\n")
    # firered: u1 one sub (2-char ref → cer 0.5)
    fr = {"u1": "巴十", "u2": "瓜娃子", "u3": "要得", "u4": "巴适"}
    with (hyps / "firered.jsonl").open("w", encoding="utf-8") as f:
        for uid, _s, _ref in rows:
            t = fr[uid]
            f.write(json.dumps({"utt_id": uid, "norm": t, "raw": t}) + "\n")
    # qwen3: u2 empty
    with (hyps / "qwen3asr.jsonl").open("w", encoding="utf-8") as f:
        for uid, _s, ref in rows:
            t = "" if uid == "u2" else ref
            f.write(json.dumps({"utt_id": uid, "norm": t, "raw": t}) + "\n")
    # vote: labels + conf spanning bins
    vote = [
        ("u1", "巴适", 1.0),
        ("u2", "瓜娃子", 0.7),
        ("u3", "要得", 0.4),
        ("u4", "巴适", 0.95),
    ]
    with (merged / "vote_P0.jsonl").open("w", encoding="utf-8") as f:
        for uid, lab, conf in vote:
            f.write(json.dumps({"utt_id": uid, "label": lab, "conf": conf, "split": "x"}) + "\n")

    summary = run_cer(man, out, systems=("sensevoice", "firered", "qwen3asr", "vote_P0"))
    ov = summary["overall"]
    assert ov["sensevoice"]["All"]["n"] == 4
    assert ov["sensevoice"]["All"]["cer"] == 0.0
    assert ov["qwen3asr"]["All"]["n_empty_hyp"] == 1
    assert ov["qwen3asr"]["Hard"]["n_empty_hyp"] == 1
    assert ov["vote_P0"]["All"]["n_empty_hyp"] == 0
    assert ov["vote_P0"]["All"]["cer"] == 0.0
    # firered: one substitution on 2-char utt; total ref chars 2+3+2+2=9, dist=1 → 1/9
    assert ov["firered"]["All"]["distance_sum"] == 1
    assert ov["firered"]["All"]["ref_len_sum"] == 9
    bins = summary["vote_conf"]
    assert bins["<0.6"]["All"]["n"] == 1
    assert bins["0.6-0.9"]["All"]["n"] == 1
    assert bins[">=0.9"]["All"]["n"] == 2
    per = [json.loads(x) for x in (out / "metrics" / "cer_per_utt.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(per) == 16
    vote_rows = [r for r in per if r["system"] == "vote_P0"]
    assert all("conf" in r for r in vote_rows)
    sv_rows = [r for r in per if r["system"] == "sensevoice"]
    assert all("conf" not in r for r in sv_rows)
    assert (out / "metrics" / "cer_overall.csv").is_file()
    assert (out / "metrics" / "cer_vote_conf.csv").is_file()
