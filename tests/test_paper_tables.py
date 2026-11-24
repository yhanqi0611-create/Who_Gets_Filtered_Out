from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import pytest

from wscaudit.audit_adjudicate import sha256_file
from wscaudit.audit_report import delta_pass_word, holm_clause
from wscaudit.paper_tables import (
    STATS_TABLE_SHA256,
    PaperInputError,
    fmt_cer,
    fmt_ci,
    fmt_p,
    fmt_pct,
    fmt_sig,
    latex_escape,
    load_stats_table,
    or_cmp,
    run_paper_tables,
)

REPO = Path(__file__).resolve().parents[1]
# Golden values from the frozen 2026-11-03 analysis and the round-1 audit; the only hard-coded numbers allowed.
GOLD_NUTT = 8373
GOLD_B = 10000
GOLD_SEED = 20261103


def _write_stats(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["name", "split", "estimate", "ci_lo", "ci_hi", "p_raw", "p_holm", "family"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _primary_rows() -> list[dict[str, str]]:
    return [
        {"name": "cer_sensevoice", "split": "All", "estimate": "0.1656", "ci_lo": "0.16", "ci_hi": "0.17", "family": "secondary"},
        {"name": "cer_firered", "split": "All", "estimate": "0.1474", "ci_lo": "0.14", "ci_hi": "0.15", "family": "secondary"},
        {"name": "cer_qwen3asr", "split": "All", "estimate": "0.1340", "ci_lo": "0.13", "ci_hi": "0.14", "family": "secondary"},
        {"name": "cer_vote_P0", "split": "All", "estimate": "0.1541", "ci_lo": "0.15", "ci_hi": "0.16", "family": "secondary"},
        {"name": "cer_vote_S2", "split": "All", "estimate": "0.1696", "ci_lo": "0.16", "ci_hi": "0.17", "family": "secondary"},
        {"name": "delta_cer_P0_minus_qwen3asr", "split": "All", "estimate": "0.0201", "ci_lo": "0.018", "ci_hi": "0.022", "p_raw": "0.0001", "p_holm": "0.0006", "family": "primary"},
        {"name": "delta_cer_P0_minus_S2", "split": "All", "estimate": "-0.0155", "ci_lo": "-0.017", "ci_hi": "-0.014", "p_raw": "0.0001", "p_holm": "0.0006", "family": "primary"},
        {"name": "delta_ece_has_minus_no", "split": "All", "estimate": "0.0125", "ci_lo": "0.003", "ci_hi": "0.022", "p_raw": "0.0086", "p_holm": "0.0534", "family": "primary"},
        {"name": "delta_pp_pass_0.9", "split": "All", "estimate": "3.36", "ci_lo": "0.77", "ci_hi": "5.93", "p_raw": "0.0267", "p_holm": "0.0534", "family": "primary"},
        {"name": "delta_retention_P0_minus_S2", "split": "All", "estimate": "0.0181", "ci_lo": "0.007", "ci_hi": "0.029", "p_raw": "0.0014", "p_holm": "0.0056", "family": "primary"},
        {"name": "logit_OR_has_dialect_P0", "split": "All", "estimate": "0.843", "ci_lo": "0.72", "ci_hi": "0.98", "p_raw": "0.027", "p_holm": "0.0534", "family": "primary"},
        {"name": "ece3_vote_P0", "split": "All", "estimate": "0.123", "ci_lo": "0.12", "ci_hi": "0.13", "family": "secondary"},
        {"name": "ece3_vote_S2", "split": "All", "estimate": "0.140", "ci_lo": "0.14", "ci_hi": "0.14", "family": "secondary"},
        {"name": "spearman_conf_cer_vote_P0", "split": "All", "estimate": "-0.582", "ci_lo": "-0.60", "ci_hi": "-0.56", "family": "secondary"},
        {"name": "spearman_conf_cer_vote_S2", "split": "All", "estimate": "-0.598", "ci_lo": "-0.62", "ci_hi": "-0.58", "family": "secondary"},
    ]


def _expand_splits(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        out.append(r)
        if r["split"] == "All" and r["name"].startswith(("cer_", "ece3_", "spearman_")):
            for sp in ("Easy", "Hard"):
                d = dict(r)
                d["split"] = sp
                out.append(d)
    return out


def _mini_findings() -> list[dict]:
    return json.loads((REPO / "results" / "audit_findings_numbers.json").read_text(encoding="utf-8"))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    results = tmp_path / "results"
    paper = tmp_path / "paper"
    results.mkdir()
    (results / "figures").mkdir()
    src_pdf = REPO / "results" / "figures" / "audit_precision_recall.pdf"
    (results / "figures" / "audit_precision_recall.pdf").write_bytes(src_pdf.read_bytes())
    (results / "audit_table.tex").write_bytes((REPO / "results" / "audit_table.tex").read_bytes())
    (results / "audit_findings_numbers.json").write_bytes((REPO / "results" / "audit_findings_numbers.json").read_bytes())
    (results / "AUDIT_REPORT_PROVENANCE.txt").write_bytes((REPO / "results" / "AUDIT_REPORT_PROVENANCE.txt").read_bytes())
    stats = tmp_path / "stats_table.csv"
    _write_stats(stats, _expand_splits(_primary_rows()))
    summary = tmp_path / "stats_summary.json"
    summary.write_text(json.dumps({"n": 8373, "B": 10000, "seed": 20261103, "rows": []}) + "\n", encoding="utf-8")
    return stats, summary, results, paper


def test_formatters() -> None:
    assert fmt_cer(0.02011, signed=True) == "+0.020"
    assert fmt_cer(-0.01549, signed=True) == "-0.015"
    assert fmt_cer(0.13398) == "0.134"
    assert fmt_sig(0.012496) == "0.0125"
    assert fmt_pct(0.71875) == "71.9"
    assert fmt_ci(0.546, 0.844, kind="pct") == "[54.6, 84.4]"
    assert fmt_p(0.0001, B=10000) == r"$<10^{-4}$"
    assert fmt_p(0.0534, B=10000) == "0.0534"
    assert latex_escape("71.9%") == r"71.9\%"
    assert "_" in latex_escape("a_b")


def test_wording_functions() -> None:
    assert delta_pass_word(3.36) == "more"
    assert or_cmp(0.843) == r"$<1$"
    assert holm_clause(0.0534, 0.0534) == "neither is significant after Holm"


def test_hash_mismatch_message(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    with pytest.raises(PaperInputError) as exc:
        run_paper_tables(
            repo_root=REPO,
            stats_table=stats,
            stats_summary=summary,
            results_dir=results,
            paper_dir=paper,
            expected_stats_sha256="deadbeef",
        )
    msg = str(exc.value)
    assert "deadbeef" in msg
    assert sha256_file(stats) in msg


def test_missing_file(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    missing = tmp_path / "nope.csv"
    with pytest.raises(PaperInputError) as exc:
        run_paper_tables(
            repo_root=REPO,
            stats_table=missing,
            stats_summary=summary,
            results_dir=results,
            paper_dir=paper,
            expected_stats_sha256=sha256_file(stats),
        )
    assert "missing file" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_duplicate_name_split(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    rows = _expand_splits(_primary_rows())
    rows.append(dict(rows[0]))
    _write_stats(stats, rows)
    with pytest.raises(PaperInputError) as exc:
        run_paper_tables(
            repo_root=REPO,
            stats_table=stats,
            stats_summary=summary,
            results_dir=results,
            paper_dir=paper,
            expected_stats_sha256=sha256_file(stats),
        )
    assert "duplicate" in str(exc.value)


def test_missing_primary_row(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    rows = [r for r in _expand_splits(_primary_rows()) if r["name"] != "logit_OR_has_dialect_P0"]
    _write_stats(stats, rows)
    with pytest.raises(PaperInputError) as exc:
        run_paper_tables(
            repo_root=REPO,
            stats_table=stats,
            stats_summary=summary,
            results_dir=results,
            paper_dir=paper,
            expected_stats_sha256=sha256_file(stats),
        )
    assert "logit_OR_has_dialect_P0" in str(exc.value)


def test_twice_byte_identical(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    h = sha256_file(stats)
    run_paper_tables(
        repo_root=REPO,
        stats_table=stats,
        stats_summary=summary,
        results_dir=results,
        paper_dir=paper,
        expected_stats_sha256=h,
    )
    first = {p: p.read_bytes() for p in paper.rglob("*") if p.is_file()}
    run_paper_tables(
        repo_root=REPO,
        stats_table=stats,
        stats_summary=summary,
        results_dir=results,
        paper_dir=paper,
        expected_stats_sha256=h,
    )
    second = {p: p.read_bytes() for p in paper.rglob("*") if p.is_file()}
    assert first == second


def test_macro_names_letters_only_and_unique(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    run_paper_tables(
        repo_root=REPO,
        stats_table=stats,
        stats_summary=summary,
        results_dir=results,
        paper_dir=paper,
        expected_stats_sha256=sha256_file(stats),
    )
    tex = (paper / "numbers.tex").read_text(encoding="utf-8")
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)
    assert names
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)


def test_repo_audit_hashes() -> None:
    prov = (REPO / "results" / "AUDIT_REPORT_PROVENANCE.txt").read_text(encoding="utf-8")
    kv = dict(line.split("=", 1) for line in prov.splitlines() if "=" in line)
    assert sha256_file(REPO / "results" / "audit_findings_numbers.json") == kv["output_sha256_audit_findings_numbers.json"]
    assert sha256_file(REPO / "results" / "audit_table.tex") == kv["input_sha256_audit_table.tex"]


def test_audit_table_matches_e23509f() -> None:
    r = subprocess.run(
        ["git", "show", "e23509f:results/audit_table.tex"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if r.returncode != 0:
        pytest.skip(f"git history unavailable: {r.stderr.decode()[:200]}")
    assert (REPO / "results" / "audit_table.tex").read_bytes() == r.stdout


def test_tab_audit_strings_in_frozen_table(tmp_path: Path) -> None:
    stats, summary, results, paper = _fixture(tmp_path)
    run_paper_tables(
        repo_root=REPO,
        stats_table=stats,
        stats_summary=summary,
        results_dir=results,
        paper_dir=paper,
        expected_stats_sha256=sha256_file(stats),
    )
    tab = (paper / "tables" / "tab_audit.tex").read_text(encoding="utf-8")
    frozen = (REPO / "results" / "audit_table.tex").read_text(encoding="utf-8")
    for s in ("71.9\\% [54.6, 84.4]", "4.2\\% [2.9, 6.4]", "8.8\\% [3.0, 23.0]"):
        assert s in tab
        assert s in frozen


def test_golden_macros() -> None:
    frozen = REPO / "results" / "frozen" / "stats_table.csv"
    if not frozen.is_file():
        pytest.skip("results/frozen/stats_table.csv is absent")
    paper = REPO / "paper"
    if not (paper / "numbers.tex").is_file():
        pytest.skip("paper/numbers.tex not generated yet")
    tex = (paper / "numbers.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+?)\\xspace\}", tex))
    side = json.loads((paper / "numbers_sources.json").read_text(encoding="utf-8"))
    by_macro = {r["macro_or_cell"]: r for r in side if re.fullmatch(r"[A-Za-z]+", r["macro_or_cell"])}
    for name, formatted in macros.items():
        assert name in by_macro
        assert by_macro[name]["formatted"] == formatted
    assert int(macros["NUtt"]) == GOLD_NUTT
    assert int(macros["NBoot"]) == GOLD_B
    assert int(macros["BootSeed"]) == GOLD_SEED
    assert abs(float(macros["CERqwen"]) - 0.134) < 0.0005
    assert abs(float(macros["CERvoteP"]) - 0.154) < 0.0005
    assert abs(float(macros["CERvoteS"]) - 0.170) < 0.0005
    assert abs(float(macros["DeltaCERPqwen"]) - 0.020) < 0.0015
    assert abs(float(macros["DeltaCERPS"]) - (-0.015)) < 0.0015
    assert abs(float(macros["DeltaECE"]) - 0.0125) < 0.00005
    assert abs(float(macros["DeltaECEpHolm"]) - 0.0258) < 0.00005
    assert abs(float(macros["DeltaPP"]) - 3.36) < 0.005
    assert float(macros["DeltaPP"]) > 0
    assert abs(float(macros["DeltaPPpHolm"]) - 0.0534) < 0.00005
    assert abs(float(macros["ORdialect"]) - 0.843) < 0.0005
    assert macros["ORcmp"] == r"$<1$"
    assert abs(float(macros["ORpHolm"]) - 0.0534) < 0.00005
    assert macros["AuditN"] == "126"
    assert macros["AuditNCtrlDialect"] == "40"
    assert macros["AuditNCtrlLowconf"] == "20"
    assert macros["AuditNMand"] == "32"
    assert macros["AuditNOver"] == "34"
    assert macros["NRefErrorOnly"] == "1"
    assert macros["MandPrecK"] == "23" and macros["MandPrecN"] == "32"
    assert macros["MandPrec"] == "71.9" and macros["MandPrecCI"] == "[54.6, 84.4]"
    assert macros["MandRecallW"] == "4.2" and macros["MandRecallWCI"] == "[2.9, 6.4]"
    assert macros["NCtrlPop"] == "1163" and macros["MissesCtrl"] == "18" and macros["NCtrlDesign"] == "40"
    assert macros["OverPrecK"] == "3" and macros["OverPrecN"] == "34"
    assert macros["OverPrec"] == "8.8" and macros["OverPrecCI"] == "[3.0, 23.0]"
    assert macros["LowconfK"] == "19" and macros["LowconfN"] == "20"
    assert macros["LowconfRate"] == "95.0" and macros["LowconfCI"] == "[76.4, 99.1]"


def _iter_paper_text() -> list[tuple[Path, str]]:
    paper = REPO / "paper"
    if not paper.is_dir():
        pytest.skip("paper/ missing")
    out = []
    for p in paper.rglob("*"):
        if not p.is_file():
            continue
        if p.name == "PAPER_PROVENANCE.txt":
            continue
        if p.suffix.lower() in {".tex", ".bib", ".md", ".json"}:
            out.append((p, p.read_text(encoding="utf-8")))
    return out


def test_guard_forbidden_phrases() -> None:
    for p, text in _iter_paper_text():
        for bad in ("23/45", "naive", "unweighted", "less often", "bias is proven", "proves"):
            assert bad not in text, f"{bad} in {p}"


def test_guard_ci_labels() -> None:
    tab = REPO / "paper" / "tables" / "tab_audit.tex"
    if not tab.is_file():
        pytest.skip("tab_audit.tex not generated")
    text = tab.read_text(encoding="utf-8")
    assert "Precision [Wilson" in text
    assert "Recall (weighted) [bootstrap" in text
    assert "Wilson" not in text.split("Recall")[-1] or "bootstrap" in text.split("Recall")[-1]


def test_guard_section_numbers() -> None:
    sec = REPO / "paper" / "sections"
    if not sec.is_dir():
        pytest.skip("paper/sections missing")
    num = re.compile(r"\d+\.\d+|\d{2,}")
    for p in sec.glob("*.tex"):
        body = "\n".join(line for line in p.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("%"))
        body = re.sub(r"\\cite[t]?\{[^}]*\}", "", body)
        body = re.sub(r"RQ\d", "", body)
        hits = num.findall(body)
        assert hits == [], f"numeric literals in {p.name}: {hits}"


def test_main_author_and_mode() -> None:
    main = REPO / "paper" / "main.tex"
    if not main.is_file():
        pytest.skip("main.tex missing")
    text = main.read_text(encoding="utf-8")
    assert r"\author{Hanqi Yang}" in text
    assert "acceptedWithA" in text


def test_ethics_and_limitations() -> None:
    ethics = REPO / "paper" / "sections" / "ethics.tex"
    lim = REPO / "paper" / "sections" / "limitations.tex"
    if not ethics.is_file() or not lim.is_file():
        pytest.skip("section files missing")
    et = ethics.read_text(encoding="utf-8")
    lt = lim.read_text(encoding="utf-8")
    assert "Ethical considerations" in et
    for needle in ("Qwen3-ASR", "LLM-GER", "Whisper", "unreliable", r"\todo{\kappa"):
        assert needle in lt or ("kappa" in lt.lower()) or ("κ" in lt) or ("kappa" in lt)


def test_freeze_prompt16_paths() -> None:
    r = subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            "5195103",
            "--",
            "results/audit_table.tex",
            "results/audit_findings_numbers.json",
            "results/figures",
            "resources/",
            "src/wscaudit/stats/",
            "src/wscaudit/audit.py",
            "src/wscaudit/audit_adjudicate.py",
            "src/wscaudit/audit_report.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        pytest.skip(f"git unavailable: {r.stderr[:200]}")
    assert r.stdout.strip() == ""
