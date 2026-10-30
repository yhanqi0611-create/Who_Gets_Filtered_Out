from __future__ import annotations

import json
from pathlib import Path

from wscaudit.merge.vote import align_pair, present_norm, run_vote, vote_one


SCHEMA = {
    "utt_id",
    "split",
    "label",
    "conf",
    "n_voters_mean",
    "n_available",
    "sources_used",
    "char_count",
    "runtime_s",
}


def test_three_identical_conf_one() -> None:
    out = vote_one({"sensevoice": "巴适", "firered": "巴适", "qwen3asr": "巴适"})
    assert out["label"] == "巴适"
    assert out["conf"] == 1.0
    assert out["n_available"] == 3
    assert out["n_voters_mean"] == 3.0


def test_majority_wins() -> None:
    out = vote_one({"sensevoice": "巴适", "firered": "巴适", "qwen3asr": "八十"})
    assert out["label"] == "巴适"
    assert out["n_available"] == 3
    assert out["conf"] < 1.0


def test_empty_hyp_n_available_two() -> None:
    out = vote_one({"sensevoice": "瓜娃子", "firered": "瓜娃子", "qwen3asr": None})
    assert out["n_available"] == 2
    assert out["sources_used"] == ["sensevoice", "firered"]
    assert out["label"] == "瓜娃子"
    assert out["conf"] == 1.0


def test_present_norm_absent() -> None:
    assert present_norm(None) is None
    assert present_norm({"norm": ""}) is None
    assert present_norm({"norm": "   "}) is None
    assert present_norm({"raw": "x"}) is None
    assert present_norm({"norm": "巴适"}) == "巴适"


def test_align_pair_equal_length() -> None:
    a, b = align_pair("巴适", "八十")
    assert len(a) == len(b)


def test_tie_prefers_sensevoice() -> None:
    # 适 / 十 / 士 all unique at last column → SenseVoice
    out = vote_one({"sensevoice": "巴适", "firered": "八十", "qwen3asr": "巴士"})
    assert out["label"].startswith("巴")
    assert "适" in out["label"]


def test_run_vote_schema_and_idempotent(tmp_path: Path) -> None:
    man = tmp_path / "manifest.jsonl"
    hyps = tmp_path / "hyps"
    hyps.mkdir()
    rows = [
        ("u1", "Hard", "巴适"),
        ("u2", "Hard", "瓜娃子"),
        ("u3", "Easy", "要得"),
    ]
    with man.open("w", encoding="utf-8") as f:
        for uid, spl, _ in rows:
            f.write(json.dumps({"utt_id": uid, "split": spl, "wav": "x", "ref": "x", "dur": 1}) + "\n")
    for name in ("sensevoice", "firered", "qwen3asr"):
        with (hyps / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for uid, _spl, text in rows:
                norm = "" if name == "qwen3asr" and uid == "u2" else text
                f.write(json.dumps({"utt_id": uid, "model": name, "raw": norm, "norm": norm, "runtime_s": 0}) + "\n")
    out = tmp_path / "merged" / "vote_P0.jsonl"
    summary = tmp_path / "merged" / "vote_P0.summary.json"
    paths = {n: hyps / f"{n}.jsonl" for n in ("sensevoice", "firered", "qwen3asr")}
    run_vote(man, paths, out, summary, split="all")
    recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 3
    for rec in recs:
        assert SCHEMA <= set(rec)
    by_id = {r["utt_id"]: r for r in recs}
    assert by_id["u2"]["n_available"] == 2
    assert by_id["u1"]["n_available"] == 3
    meta = json.loads(summary.read_text(encoding="utf-8"))
    assert meta["n"] == 3
    assert meta["n_with_2"] == 1
    assert meta["n_with_3"] == 2
    run_vote(man, paths, out, summary, split="all")
    recs2 = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(recs2) == 3
