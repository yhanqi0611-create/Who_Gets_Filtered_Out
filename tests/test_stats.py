from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wscaudit.metrics.calibration import EQUAL_WIDTH_10, PRIMARY_BINS, ece_and_bins
from wscaudit.stats.bootstrap import bootstrap_p, micro_cer_batch, percentile_ci, stratified_indices
from wscaudit.stats.holm import holm_adjust
from wscaudit.stats.permutation import pass_delta_pp, permute_has_within_split, permutation_p
from wscaudit.stats.run import run_stats


def test_ece_bin_edges_differ() -> None:
    assert PRIMARY_BINS != EQUAL_WIDTH_10
    rows = [{"conf": x, "acc": x} for x in (0.05, 0.15, 0.55, 0.65, 0.85, 0.95)]
    e3, b3 = ece_and_bins(rows, PRIMARY_BINS)
    e10, b10 = ece_and_bins(rows, EQUAL_WIDTH_10)
    assert e3 == pytest.approx(0.0)
    assert [tuple(b[:2]) for b in ((d["bin_lo"], d["bin_hi"]) for d in b3)] != [
        (d["bin_lo"], d["bin_hi"]) for d in b10
    ]


def test_holm_hand_calc() -> None:
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.06)


def test_bootstrap_identical_delta_ci_contains_zero() -> None:
    rng = np.random.default_rng(0)
    dist = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    refl = np.ones(6)
    splits = np.array(["Easy", "Easy", "Easy", "Hard", "Hard", "Hard"], dtype=object)
    idx = stratified_indices(splits, 400, rng, split_filter=None)
    boot = micro_cer_batch(dist, refl, idx) - micro_cer_batch(dist, refl, idx)
    lo, hi = percentile_ci(boot)
    assert lo <= 0.0 <= hi
    assert bootstrap_p(boot, 400) > 0.1


def test_permutation_separated_small_p() -> None:
    has = np.array([True] * 12 + [False] * 12 + [True] * 12 + [False] * 12)
    passed = has.copy()
    splits = np.array(["Easy"] * 24 + ["Hard"] * 24, dtype=object)
    obs = pass_delta_pp(has, passed)
    assert obs == pytest.approx(100.0)
    rng = np.random.default_rng(1)
    B = 400
    null_has = permute_has_within_split(has, splits, B, rng)
    nulls = np.array([pass_delta_pp(null_has[b], passed) for b in range(B)])
    p = permutation_p(obs, nulls, B)
    assert p < 0.05


def _write_panel(tmp_path: Path, *, s2_same_as_p0: bool = True) -> Path:
    out = tmp_path / "out"
    (out / "metrics").mkdir(parents=True)
    ids = [f"u{i}" for i in range(8)]
    splits = ["Easy", "Easy", "Easy", "Easy", "Hard", "Hard", "Hard", "Hard"]
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for uid, spl, dur in zip(ids, splits, [1.0, 1.2, 0.8, 1.1, 2.0, 1.5, 1.8, 2.2]):
            f.write(json.dumps({"utt_id": uid, "split": spl, "ref": "巴适", "dur": dur, "wav": "x"}) + "\n")
    systems = ["sensevoice", "firered", "qwen3asr", "vote_P0", "vote_S2"]
    # distances: qwen best, P0 close, S2 same as P0 if flag
    dist = {
        "sensevoice": [2, 2, 1, 2, 3, 2, 2, 3],
        "firered": [2, 1, 2, 2, 2, 3, 2, 2],
        "qwen3asr": [1, 1, 0, 1, 1, 1, 2, 1],
        "vote_P0": [1, 1, 0, 1, 1, 1, 2, 1],
        "vote_S2": [1, 1, 0, 1, 1, 1, 2, 1] if s2_same_as_p0 else [2, 2, 1, 2, 2, 2, 3, 2],
    }
    refl = [4] * 8
    conf_p0 = [0.95, 0.5, 0.95, 0.5, 0.95, 0.7, 0.4, 0.92]
    conf_s2 = list(conf_p0)
    has = [True, True, False, False, True, False, False, True]
    with (out / "metrics" / "cer_per_utt.jsonl").open("w", encoding="utf-8") as f:
        for i, uid in enumerate(ids):
            for s in systems:
                rec = {
                    "utt_id": uid,
                    "split": splits[i],
                    "system": s,
                    "ref_len": refl[i],
                    "distance": dist[s][i],
                    "cer": dist[s][i] / refl[i],
                }
                if s.startswith("vote"):
                    rec["conf"] = conf_p0[i] if s == "vote_P0" else conf_s2[i]
                f.write(json.dumps(rec) + "\n")
    with (out / "metrics" / "markers_per_utt.jsonl").open("w", encoding="utf-8") as f:
        for i, uid in enumerate(ids):
            for s, conf in (("vote_P0", conf_p0[i]), ("vote_S2", conf_s2[i])):
                f.write(
                    json.dumps(
                        {
                            "utt_id": uid,
                            "split": splits[i],
                            "system": s,
                            "conf": conf,
                            "ref_dialect": ["啥子"] if has[i] else [],
                            "hyp_dialect": ["啥子"] if has[i] else [],
                            "n_ref_dialect": 1 if has[i] else 0,
                            "n_retained": 1 if has[i] else 0,
                            "has_dialect_ref": has[i],
                            "pass_0.9": conf >= 0.9,
                            "pass_0.6": conf >= 0.6,
                            "retained_token_frac": 1.0 if has[i] else None,
                            "mandarinization_n": 0,
                            "over_dialect_n": 0,
                        }
                    )
                    + "\n"
                )
    return out


def test_seed_reproducible(tmp_path: Path) -> None:
    out = _write_panel(tmp_path)
    a = run_stats(out, out / "metrics", expected_n=8, B=20, seed=20261103)
    b = run_stats(out, out / "metrics", expected_n=8, B=20, seed=20261103)
    ra = { (r["name"], r["split"]): r["estimate"] for r in a["rows"] }
    rb = { (r["name"], r["split"]): r["estimate"] for r in b["rows"] }
    assert ra == rb
    ca = { (r["name"], r["split"]): (r["ci_lo"], r["ci_hi"]) for r in a["rows"] }
    cb = { (r["name"], r["split"]): (r["ci_lo"], r["ci_hi"]) for r in b["rows"] }
    assert ca == cb


def test_run_stats_identical_s2_delta_ci(tmp_path: Path) -> None:
    out = _write_panel(tmp_path, s2_same_as_p0=True)
    summary = run_stats(out, out / "metrics", expected_n=8, B=30, seed=1)
    row = next(r for r in summary["rows"] if r["name"] == "delta_cer_P0_minus_S2" and r["split"] == "All")
    assert row["estimate"] == pytest.approx(0.0)
    assert row["ci_lo"] <= 0.0 <= row["ci_hi"]
    assert (out / "metrics" / "stats_report.md").is_file()
    assert (out / "metrics" / "stats_table.csv").is_file()
