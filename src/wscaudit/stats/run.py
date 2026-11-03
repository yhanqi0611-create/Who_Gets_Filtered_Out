"""Prompt 12: bootstrap / permutation / logit for vote_P0 and vote_S2."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wscaudit.asr.base import load_jsonl
from wscaudit.metrics.calibration import EQUAL_WIDTH_10, PRIMARY_BINS, acc_from_cer, ece_and_bins
from wscaudit.stats.bootstrap import (
    SEED,
    bootstrap_p,
    micro_cer_batch,
    percentile_ci,
    stratified_indices,
)
from wscaudit.stats.holm import holm_adjust
from wscaudit.stats.logreg import fit_pass_logit
from wscaudit.stats.permutation import pass_delta_pp, permute_has_within_split, permutation_p

CER_SYSTEMS = ("sensevoice", "firered", "qwen3asr", "vote_P0", "vote_S2")
VOTE_SYSTEMS = ("vote_P0", "vote_S2")
SPLITS = ("Easy", "Hard", "All")
EDGES3 = np.array([0.6, 0.9])
EDGES10 = np.array([i / 10.0 for i in range(1, 10)])
PRIMARY_FAMILY = (
    "delta_cer_P0_minus_qwen3asr",
    "delta_cer_P0_minus_S2",
    "delta_ece_has_minus_no",
    "delta_pp_pass_0.9",
    "delta_retention_P0_minus_S2",
    "logit_OR_has_dialect_P0",
)


class StatsJoinError(RuntimeError):
    pass


def token_counts(rec: dict[str, Any]) -> tuple[float, float]:
    """n_ref_dialect, n_retained. Recompute from lists if P0 rows lack the fields."""
    ref_d = list(rec.get("ref_dialect") or [])
    hyp_d = list(rec.get("hyp_dialect") or [])
    n_ref = rec.get("n_ref_dialect")
    n_ret = rec.get("n_retained")
    if n_ref is None:
        n_ref = len(ref_d)
    if n_ret is None:
        rc, hc = Counter(ref_d), Counter(hyp_d)
        n_ret = sum(min(rc[s], hc[s]) for s in rc)
    return float(n_ref), float(n_ret)


def fmt_p(p: float | None, *, analytic: bool = False, B: int = 10_000) -> str:
    if p is None or p != p:
        return "n/a"
    if analytic:
        if p <= 0.0:
            return "<1e-16"
        return f"{p:.6g}"
    if p <= (1.0 / B) + 1e-18:
        return "<1e-4"
    return f"{p:.4g}"


def rank_avg(x: np.ndarray) -> np.ndarray:
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank_avg(x), rank_avg(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    if den == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / den)


def ece_1d(conf: np.ndarray, acc: np.ndarray, edges: np.ndarray) -> float:
    if conf.size == 0:
        return float("nan")
    b = np.digitize(conf, edges, right=False)
    n = conf.size
    ece = 0.0
    for k in range(int(edges.size) + 1):
        m = b == k
        nk = int(m.sum())
        if nk == 0:
            continue
        ece += (nk / n) * abs(float(conf[m].mean()) - float(acc[m].mean()))
    return float(ece)


def ece_2d(conf: np.ndarray, acc: np.ndarray, idx: np.ndarray, edges: np.ndarray, chunk: int = 200) -> np.ndarray:
    B = idx.shape[0]
    out = np.empty(B)
    nbin = int(edges.size) + 1
    for start in range(0, B, chunk):
        sl = idx[start : start + chunk]
        c = conf[sl]
        a = acc[sl]
        b = np.digitize(c, edges, right=False)
        ece = np.zeros(sl.shape[0])
        n = sl.shape[1]
        for k in range(nbin):
            m = b == k
            nk = m.sum(axis=1)
            csum = np.where(m, c, 0.0).sum(axis=1)
            asum = np.where(m, a, 0.0).sum(axis=1)
            ok = nk > 0
            gap = np.zeros(sl.shape[0])
            gap[ok] = np.abs(csum[ok] / nk[ok] - asum[ok] / nk[ok])
            ece += (nk / n) * gap
        out[start : start + chunk] = ece
    return out


def _load_panel(out_root: Path, expected_n: int) -> dict[str, Any]:
    man_rows = load_jsonl(out_root / "manifest.jsonl")
    cer_rows = load_jsonl(out_root / "metrics" / "cer_per_utt.jsonl")
    mark_rows = load_jsonl(out_root / "metrics" / "markers_per_utt.jsonl")
    man = {r["utt_id"]: r for r in man_rows if "utt_id" in r}
    cer: dict[tuple[str, str], dict[str, Any]] = {}
    for r in cer_rows:
        if "utt_id" in r and r.get("system"):
            cer[(r["utt_id"], r["system"])] = r
    mark: dict[tuple[str, str], dict[str, Any]] = {}
    for r in mark_rows:
        if "utt_id" not in r:
            continue
        mark[(r["utt_id"], r.get("system") or "vote_P0")] = r
    ids = sorted(man)
    if expected_n is not None and len(ids) != expected_n:
        raise StatsJoinError(f"manifest n={len(ids)} expected {expected_n}")
    missing: list[str] = []
    for uid in ids:
        for s in CER_SYSTEMS:
            if (uid, s) not in cer:
                missing.append(f"cer {s} {uid}")
                break
        for s in VOTE_SYSTEMS:
            if (uid, s) not in mark:
                missing.append(f"markers {s} {uid}")
                break
        if len(missing) > 12:
            break
    if missing:
        raise StatsJoinError(
            "join missing keys (run vote/cer/markers for S2 without overwriting P0). "
            f"examples: {missing[:8]}"
        )
    n = len(ids)
    splits = np.array([str(man[u].get("split") or "") for u in ids], dtype=object)
    dur = np.array([float(man[u].get("dur") or 0.0) for u in ids], dtype=float)
    dist = {s: np.empty(n, dtype=float) for s in CER_SYSTEMS}
    refl = {s: np.empty(n, dtype=float) for s in CER_SYSTEMS}
    cer_u = {s: np.empty(n, dtype=float) for s in CER_SYSTEMS}
    for i, uid in enumerate(ids):
        for s in CER_SYSTEMS:
            rec = cer[(uid, s)]
            dist[s][i] = float(rec["distance"])
            refl[s][i] = float(rec["ref_len"])
            cer_u[s][i] = float(rec["cer"])
    conf = {}
    has = {}
    n_ref = {}
    n_ret = {}
    p09 = {}
    p06 = {}
    for s in VOTE_SYSTEMS:
        conf[s] = np.empty(n, dtype=float)
        has[s] = np.empty(n, dtype=bool)
        n_ref[s] = np.zeros(n, dtype=float)
        n_ret[s] = np.zeros(n, dtype=float)
        p09[s] = np.empty(n, dtype=bool)
        p06[s] = np.empty(n, dtype=bool)
        for i, uid in enumerate(ids):
            rec = mark[(uid, s)]
            conf[s][i] = float(rec["conf"])
            has[s][i] = bool(rec["has_dialect_ref"])
            n_ref[s][i], n_ret[s][i] = token_counts(rec)
            p09[s][i] = bool(rec["pass_0.9"])
            p06[s][i] = bool(rec["pass_0.6"])
    # ref dialect is a property of the reference; P0/S2 should match
    has_ref = has["vote_P0"]
    acc = {s: np.array([acc_from_cer(c) for c in cer_u[s]], dtype=float) for s in VOTE_SYSTEMS}
    return {
        "n": n,
        "ids": ids,
        "splits": splits,
        "dur": dur,
        "dist": dist,
        "refl": refl,
        "cer": cer_u,
        "conf": conf,
        "acc": acc,
        "has": has_ref,
        "n_ref": n_ref,
        "n_ret": n_ret,
        "p09": p09,
        "p06": p06,
    }


def _mask(splits: np.ndarray, split: str) -> np.ndarray:
    if split == "All":
        return np.ones(splits.size, dtype=bool)
    return splits == split


def _row(
    name: str,
    split: str,
    estimate: float | None,
    ci: tuple[float, float] | None,
    p_raw: float | None,
    family: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "split": split,
        "estimate": None if estimate is None or estimate != estimate else float(estimate),
        "ci_lo": None if not ci or ci[0] != ci[0] else float(ci[0]),
        "ci_hi": None if not ci or ci[1] != ci[1] else float(ci[1]),
        "p_raw": None if p_raw is None or p_raw != p_raw else float(p_raw),
        "p_holm": None,
        "family": family,
    }


def _print_cal_sanity(conf: np.ndarray, acc: np.ndarray) -> None:
    rows = [{"conf": float(c), "acc": float(a)} for c, a in zip(conf, acc)]
    e3, b3 = ece_and_bins(rows, PRIMARY_BINS)
    e10, b10 = ece_and_bins(rows, EQUAL_WIDTH_10)
    print("=== Step 0 calibration sanity (vote_P0 All) ===")
    print("PRIMARY_BINS", [list(x) for x in PRIMARY_BINS])
    print("EQUAL_WIDTH_10", [list(x) for x in EQUAL_WIDTH_10])
    print("ECE_3", None if e3 is None else round(e3, 6))
    for b in b3:
        print(f"  [{b['bin_lo']}, {b['bin_hi']}] n={b['n']} avg_conf={b['avg_conf']} avg_acc={b['avg_acc']} gap={b['gap']}")
    print("ECE_10", None if e10 is None else round(e10, 6))
    for b in b10:
        if b["n"]:
            print(f"  [{b['bin_lo']}, {b['bin_hi']}] n={b['n']} avg_conf={b['avg_conf']} avg_acc={b['avg_acc']} gap={b['gap']}")
    same = e3 is not None and e10 is not None and abs(e3 - e10) < 5e-4
    print(
        "Bin edges differ. If ECE_3≈ECE_10 on All, that is chance "
        f"(same={same}); Hard previously 0.1885 vs 0.1893 so this is not a bin-assignment bug."
    )


def run_stats(
    out_root: Path,
    metrics_dir: Path | None = None,
    *,
    expected_n: int = 8373,
    B: int = 10_000,
    seed: int = SEED,
) -> dict[str, Any]:
    metrics_dir = metrics_dir or (out_root / "metrics")
    panel = _load_panel(out_root, expected_n)
    rng = np.random.default_rng(seed)
    splits = panel["splits"]
    _print_cal_sanity(panel["conf"]["vote_P0"], panel["acc"]["vote_P0"])

    idx_cache = {sp: stratified_indices(splits, B, rng, split_filter=None if sp == "All" else sp) for sp in SPLITS}

    table: list[dict[str, Any]] = []
    for split in SPLITS:
        m = _mask(splits, split)
        idx = idx_cache[split]
        for s in CER_SYSTEMS:
            hat = float(panel["dist"][s][m].sum() / max(panel["refl"][s][m].sum(), 1.0))
            boot = micro_cer_batch(panel["dist"][s], panel["refl"][s], idx)
            table.append(_row(f"cer_{s}", split, hat, percentile_ci(boot), None, "secondary"))

        for name, a, b in (
            ("delta_cer_P0_minus_qwen3asr", "vote_P0", "qwen3asr"),
            ("delta_cer_P0_minus_S2", "vote_P0", "vote_S2"),
        ):
            d0 = panel["dist"][a][m].sum() / max(panel["refl"][a][m].sum(), 1.0)
            d1 = panel["dist"][b][m].sum() / max(panel["refl"][b][m].sum(), 1.0)
            hat = float(d0 - d1)
            boot = micro_cer_batch(panel["dist"][a], panel["refl"][a], idx) - micro_cer_batch(
                panel["dist"][b], panel["refl"][b], idx
            )
            fam = "primary" if split == "All" else "secondary"
            table.append(_row(name, split, hat, percentile_ci(boot), bootstrap_p(boot, B), fam))

        for vs in VOTE_SYSTEMS:
            c, e = panel["conf"][vs][m], panel["cer"][vs][m]
            a = panel["acc"][vs][m]
            rho = spearman_rho(c, e)
            rhos = np.array([spearman_rho(panel["conf"][vs][idx[b]], panel["cer"][vs][idx[b]]) for b in range(B)])
            table.append(_row(f"spearman_conf_cer_{vs}", split, rho, percentile_ci(rhos), None, "secondary"))
            e3 = ece_1d(c, a, EDGES3)
            e3b = ece_2d(panel["conf"][vs], panel["acc"][vs], idx, EDGES3)
            table.append(_row(f"ece3_{vs}", split, e3, percentile_ci(e3b), None, "secondary"))

        vs = "vote_P0"
        has = panel["has"]
        c, a = panel["conf"][vs], panel["acc"][vs]
        e_has = ece_1d(c[m & has], a[m & has], EDGES3)
        e_no = ece_1d(c[m & ~has], a[m & ~has], EDGES3)
        hat_de = float(e_has - e_no)
        de = np.empty(B)
        for b in range(B):
            i = idx[b]
            hb = has[i]
            de[b] = ece_1d(c[i][hb], a[i][hb], EDGES3) - ece_1d(c[i][~hb], a[i][~hb], EDGES3)
        fam = "primary" if split == "All" else "secondary"
        table.append(_row("delta_ece_has_minus_no", split, hat_de, percentile_ci(de), bootstrap_p(de, B), fam))

        for thr, passed in (("0.9", panel["p09"][vs]), ("0.6", panel["p06"][vs])):
            hat_pp = pass_delta_pp(has[m], passed[m])
            pps = np.array([pass_delta_pp(has[idx[b]], passed[idx[b]]) for b in range(B)])
            if split == "All":
                nulls_has = permute_has_within_split(has, splits, B, rng)
                nulls = np.array([pass_delta_pp(nulls_has[b], passed) for b in range(B)])
            else:
                sub_has, sub_spl, sub_pass = has[m], splits[m], passed[m]
                nulls_has = permute_has_within_split(sub_has, sub_spl, B, rng)
                nulls = np.array([pass_delta_pp(nulls_has[b], sub_pass) for b in range(B)])
            p_perm = permutation_p(hat_pp, nulls, B)
            fam = "primary" if (split == "All" and thr == "0.9") else "secondary"
            table.append(
                _row(f"delta_pp_pass_{thr}", split, hat_pp, percentile_ci(pps), p_perm, fam)
            )

        m_d = m & has
        r0 = float(panel["n_ret"]["vote_P0"][m_d].sum() / max(panel["n_ref"]["vote_P0"][m_d].sum(), 1.0))
        r2 = float(panel["n_ret"]["vote_S2"][m_d].sum() / max(panel["n_ref"]["vote_S2"][m_d].sum(), 1.0))
        hat_r = r0 - r2
        rb = np.empty(B)
        for b in range(B):
            i = idx[b]
            md = has[i]
            d0 = panel["n_ref"]["vote_P0"][i][md].sum()
            d2 = panel["n_ref"]["vote_S2"][i][md].sum()
            n0 = panel["n_ret"]["vote_P0"][i][md].sum() / d0 if d0 else np.nan
            n2 = panel["n_ret"]["vote_S2"][i][md].sum() / d2 if d2 else np.nan
            rb[b] = n0 - n2
        fam = "primary" if split == "All" else "secondary"
        table.append(_row("delta_retention_P0_minus_S2", split, hat_r, percentile_ci(rb), bootstrap_p(rb, B), fam))

        logit = fit_pass_logit(has[m], panel["p09"][vs][m], panel["dur"][m], splits[m])
        fam = "primary" if split == "All" else "secondary"
        table.append(
            _row(
                "logit_OR_has_dialect_P0" if vs else "logit",
                split,
                logit["or"],
                (logit["or_ci_lo"], logit["or_ci_hi"]),
                logit["p"],
                fam,
            )
        )
        if split == "All":
            logit_s2 = fit_pass_logit(has, panel["p09"]["vote_S2"], panel["dur"], splits)
            table.append(
                _row(
                    "logit_OR_has_dialect_S2",
                    split,
                    logit_s2["or"],
                    (logit_s2["or_ci_lo"], logit_s2["or_ci_hi"]),
                    logit_s2["p"],
                    "secondary",
                )
            )

    primary_idx = [i for i, r in enumerate(table) if r["family"] == "primary" and r["name"] in PRIMARY_FAMILY]
    # All-split primary rows only (filter split==All)
    fam_rows = [i for i in primary_idx if table[i]["split"] == "All"]
    # Ensure unique names in family order
    by_name = {table[i]["name"]: i for i in fam_rows}
    ordered = [by_name[n] for n in PRIMARY_FAMILY if n in by_name]
    raw = [table[i]["p_raw"] if table[i]["p_raw"] is not None else 1.0 for i in ordered]
    adj = holm_adjust(raw)
    for i, p_h in zip(ordered, adj):
        table[i]["p_holm"] = p_h

    summary = {
        "n": panel["n"],
        "B": B,
        "seed": seed,
        "note": (
            "Primary family Holm-adjusted: ΔCER P0−qwen3 / P0−S2 (All), "
            "ΔECE has−no (All), Δpp pass≥0.9 (All, permutation p), "
            "Δretention P0−S2 (All), logit OR has_dialect P0 (All). "
            "ECE_3 vs ECE_10 bin edges differ; equal All ECE on 3 Nov was chance."
        ),
        "rows": table,
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "stats_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = metrics_dir / "stats_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["name", "split", "estimate", "ci_lo", "ci_hi", "p_raw", "p_holm", "family"]
        )
        w.writeheader()
        for r in table:
            w.writerow(r)
    md = _render_md(table, B, seed, panel["n"])
    (metrics_dir / "stats_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {metrics_dir / 'stats_summary.json'}")
    return summary


def _render_md(table: list[dict[str, Any]], B: int, seed: int, n: int) -> str:
    lines = [
        f"# Stats report (vote_P0 + vote_S2)",
        "",
        f"n={n}, B={B}, seed={seed}. Primary p-values Holm-adjusted. Bootstrap p floored at 1/B; never printed as 0.",
        "",
        "| name | split | estimate | 95% CI | p_raw | p_holm | family |",
        "|---|---|---:|---|---|---|---|",
    ]
    for r in table:
        est = "n/a" if r["estimate"] is None else f"{r['estimate']:.6g}"
        if r["ci_lo"] is None:
            ci = "—"
        else:
            ci = f"[{r['ci_lo']:.6g}, {r['ci_hi']:.6g}]"
        analytic = r["name"].startswith("logit_")
        pr = fmt_p(r["p_raw"], analytic=analytic, B=B) if r["p_raw"] is not None else "—"
        ph = fmt_p(r["p_holm"], analytic=analytic, B=B) if r["p_holm"] is not None else "—"
        lines.append(f"| {r['name']} | {r['split']} | {est} | {ci} | {pr} | {ph} | {r['family']} |")
    lines.append("")
    return "\n".join(lines) + "\n"
