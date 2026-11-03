"""Calibration of vote_P0 confidence (Prompt 10 / RQ2). CPU only.

Accuracy proxy: ``acc = 1 - min(cer, 1.0)``. Per-utterance CER can exceed 1
when insertions dominate; it is clipped **only** for ECE / reliability.

Primary ECE uses the same three edges as the CER table:
``[0.0, 0.6)``, ``[0.6, 0.9)``, ``[0.9, 1.0]`` (1.0 included in the last bin).
Ten equal-width bins are a sensitivity check, not the primary number.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from scipy.stats import spearmanr

from wscaudit.asr.base import load_jsonl

PRIMARY_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.6), (0.6, 0.9), (0.9, 1.0))
EQUAL_WIDTH_10: tuple[tuple[float, float], ...] = tuple(
    (i / 10.0, (i + 1) / 10.0) for i in range(10)
)
EXPECTED_N = 8373
VOTE_SYSTEM = "vote_P0"


class CalibrationJoinError(RuntimeError):
    """Raised when vote / CER / marker keys do not line up."""


def acc_from_cer(cer: float) -> float:
    return 1.0 - min(float(cer), 1.0)


def _bin_index(conf: float, bins: Sequence[tuple[float, float]]) -> int:
    c = float(conf)
    last = len(bins) - 1
    for i, (lo, hi) in enumerate(bins):
        if i == last:
            if c >= lo:
                return i
        elif lo <= c < hi:
            return i
    return 0 if c < bins[0][0] else last


def ece_and_bins(
    rows: Sequence[dict[str, Any]],
    bins: Sequence[tuple[float, float]] = PRIMARY_BINS,
) -> tuple[float | None, list[dict[str, Any]]]:
    n = len(rows)
    buckets: list[list[dict[str, Any]]] = [[] for _ in bins]
    for row in rows:
        buckets[_bin_index(row["conf"], bins)].append(row)
    details: list[dict[str, Any]] = []
    ece = 0.0
    for (lo, hi), bucket in zip(bins, buckets):
        nk = len(bucket)
        if nk == 0:
            details.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "n": 0,
                    "avg_conf": None,
                    "avg_acc": None,
                    "gap": None,
                }
            )
            continue
        avg_conf = sum(float(r["conf"]) for r in bucket) / nk
        avg_acc = sum(float(r["acc"]) for r in bucket) / nk
        gap = abs(avg_conf - avg_acc)
        ece += (nk / n) * gap
        details.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": nk,
                "avg_conf": round(avg_conf, 6),
                "avg_acc": round(avg_acc, 6),
                "gap": round(gap, 6),
            }
        )
    if n == 0:
        return None, details
    return ece, details


def spearman_pair(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float | None]:
    if len(xs) < 2:
        return {"rho": None, "pvalue": None}
    res = spearmanr(xs, ys)
    rho = float(res.statistic) if res.statistic == res.statistic else None
    p = float(res.pvalue) if res.pvalue == res.pvalue else None
    return {
        "rho": None if rho is None else round(rho, 6),
        "pvalue": None if p is None else float(p),
    }


def join_rows(
    vote_path: Path,
    cer_path: Path,
    markers_path: Path,
    *,
    expected_n: int | None = EXPECTED_N,
    system: str = VOTE_SYSTEM,
) -> list[dict[str, Any]]:
    votes = {r["utt_id"]: r for r in load_jsonl(vote_path) if "utt_id" in r}
    cers = {
        r["utt_id"]: r
        for r in load_jsonl(cer_path)
        if r.get("system") == system and "utt_id" in r
    }
    marks = {r["utt_id"]: r for r in load_jsonl(markers_path) if "utt_id" in r}
    missing: list[str] = []
    extra_vote = sorted(set(votes) - set(cers) | set(votes) - set(marks))
    extra_cer = sorted(set(cers) - set(votes) | set(cers) - set(marks))
    extra_mark = sorted(set(marks) - set(votes) | set(marks) - set(cers))
    if extra_vote:
        missing.append(f"vote keys missing from cer/markers ({len(extra_vote)}), e.g. {extra_vote[:5]}")
    if extra_cer:
        missing.append(f"cer keys missing from vote/markers ({len(extra_cer)}), e.g. {extra_cer[:5]}")
    if extra_mark:
        missing.append(f"marker keys missing from vote/cer ({len(extra_mark)}), e.g. {extra_mark[:5]}")
    ids = set(votes) & set(cers) & set(marks)
    if missing or (expected_n is not None and len(ids) != expected_n):
        got = len(ids)
        parts = "; ".join(missing) if missing else "sets aligned"
        raise CalibrationJoinError(
            f"calibration join failed: n={got} expected_n={expected_n}; {parts}"
        )
    rows: list[dict[str, Any]] = []
    for utt_id in sorted(ids):
        v, c, m = votes[utt_id], cers[utt_id], marks[utt_id]
        if v.get("conf") is None:
            raise CalibrationJoinError(f"missing conf for {utt_id}")
        if c.get("cer") is None:
            raise CalibrationJoinError(f"missing cer for {utt_id}")
        if "has_dialect_ref" not in m:
            raise CalibrationJoinError(f"missing has_dialect_ref for {utt_id}")
        cer = float(c["cer"])
        split = str(v.get("split") or c.get("split") or m.get("split") or "")
        rows.append(
            {
                "utt_id": utt_id,
                "split": split,
                "conf": float(v["conf"]),
                "cer": cer,
                "acc": acc_from_cer(cer),
                "has_dialect_ref": bool(m["has_dialect_ref"]),
                "distance": c.get("distance"),
                "ref_len": c.get("ref_len"),
            }
        )
    return rows


def _summarise_group(name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    confs = [float(r["conf"]) for r in rows]
    cers = [float(r["cer"]) for r in rows]
    accs = [float(r["acc"]) for r in rows]
    ece3, bins3 = ece_and_bins(rows, PRIMARY_BINS)
    ece10, bins10 = ece_and_bins(rows, EQUAL_WIDTH_10)
    return {
        "n": len(rows),
        "spearman_conf_cer": spearman_pair(confs, cers),
        "spearman_conf_acc": spearman_pair(confs, accs),
        "ece_primary": {
            "ece": None if ece3 is None else round(ece3, 6),
            "bins": bins3,
            "edges": [list(b) for b in PRIMARY_BINS],
        },
        "ece_equal_width_10": {
            "ece": None if ece10 is None else round(ece10, 6),
            "bins": bins10,
            "note": "sensitivity, not primary",
        },
        "group": name,
    }


def _plot_reliability(bins: Sequence[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [b["avg_conf"] for b in bins if b["n"]]
    ys = [b["avg_acc"] for b in bins if b["n"]]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="y = x")
    if xs:
        ax.plot(xs, ys, marker="o", color="C0", label="bin mean")
        for b in bins:
            if b["n"]:
                ax.annotate(str(b["n"]), (b["avg_conf"], b["avg_acc"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("mean acc (1 − min(CER, 1))")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_line(label: str, block: dict[str, Any]) -> None:
    sc = block["spearman_conf_cer"]
    sa = block["spearman_conf_acc"]
    ece = block["ece_primary"]["ece"]
    e10 = block["ece_equal_width_10"]["ece"]

    def _f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    print(
        f"{label:<22} n={block['n']:<5} "
        f"ρ(conf,cer)={_f(sc['rho'])} p={_f(sc['pvalue'])}  "
        f"ρ(conf,acc)={_f(sa['rho'])}  "
        f"ECE_3={_f(ece)}  ECE_10={_f(e10)}"
    )


def run_calibration(
    vote_path: Path,
    cer_path: Path,
    markers_path: Path,
    metrics_dir: Path,
    *,
    expected_n: int | None = EXPECTED_N,
    system: str = VOTE_SYSTEM,
    plot: bool = True,
) -> dict[str, Any]:
    rows = join_rows(
        vote_path, cer_path, markers_path, expected_n=expected_n, system=system
    )
    groups: dict[str, list[dict[str, Any]]] = {
        "All": rows,
        "Easy": [r for r in rows if r["split"] == "Easy"],
        "Hard": [r for r in rows if r["split"] == "Hard"],
        "All_has_dialect_ref": [r for r in rows if r["has_dialect_ref"]],
        "All_no_dialect_ref": [r for r in rows if not r["has_dialect_ref"]],
    }
    summarised = {name: _summarise_group(name, rs) for name, rs in groups.items()}
    summary = {
        "n": len(rows),
        "system": system,
        "acc": "1 - min(cer, 1.0); CER may exceed 1 with insertions; clip for calibration only",
        "primary_bins": [list(b) for b in PRIMARY_BINS],
        "groups": summarised,
        "figs": {
            "All": str(metrics_dir / "figs" / "reliability_All.png"),
            "Easy": str(metrics_dir / "figs" / "reliability_Easy.png"),
            "Hard": str(metrics_dir / "figs" / "reliability_Hard.png"),
        },
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = metrics_dir / "figs"
    if plot:
        for split in ("All", "Easy", "Hard"):
            _plot_reliability(
                summarised[split]["ece_primary"]["bins"],
                fig_dir / f"reliability_{split}.png",
                f"Reliability {split} ({system})",
            )
    csv_path = metrics_dir / "calibration_bins.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "subgroup", "bin_lo", "bin_hi", "n", "avg_conf", "avg_acc", "gap"],
        )
        w.writeheader()
        mapping = [
            ("All", "all", "All"),
            ("Easy", "all", "Easy"),
            ("Hard", "all", "Hard"),
            ("All", "has_dialect_ref", "All_has_dialect_ref"),
            ("All", "no_dialect_ref", "All_no_dialect_ref"),
        ]
        for split, subgroup, key in mapping:
            for b in summarised[key]["ece_primary"]["bins"]:
                w.writerow(
                    {
                        "split": split,
                        "subgroup": subgroup,
                        "bin_lo": b["bin_lo"],
                        "bin_hi": b["bin_hi"],
                        "n": b["n"],
                        "avg_conf": b["avg_conf"],
                        "avg_acc": b["avg_acc"],
                        "gap": b["gap"],
                    }
                )
            for b in summarised[key]["ece_equal_width_10"]["bins"]:
                w.writerow(
                    {
                        "split": split,
                        "subgroup": f"{subgroup}|eq10",
                        "bin_lo": b["bin_lo"],
                        "bin_hi": b["bin_hi"],
                        "n": b["n"],
                        "avg_conf": b["avg_conf"],
                        "avg_acc": b["avg_acc"],
                        "gap": b["gap"],
                    }
                )
    sum_path = metrics_dir / "calibration_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_line("All", summarised["All"])
    _print_line("Easy", summarised["Easy"])
    _print_line("Hard", summarised["Hard"])
    _print_line("All has_dialect", summarised["All_has_dialect_ref"])
    _print_line("All no_dialect", summarised["All_no_dialect_ref"])
    print(f"wrote {sum_path}")
    print(f"wrote {csv_path}")
    if plot:
        print(f"wrote figs under {fig_dir}")
    return summary
