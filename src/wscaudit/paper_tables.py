"""Generate TACL tables and number macros from frozen analysis files (Prompt 17)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from wscaudit.audit_adjudicate import sha256_file
from wscaudit.audit_report import Trace, delta_pass_word, git_head, holm_clause

_REPO_ROOT = Path(__file__).resolve().parents[2]
STATS_TABLE_SHA256 = "b40588e666b89bf96944ef2b823e4ace7fd079c0ed413e1668988d2d3680447c"
TACL_STYLE_SOURCE = "https://transacl.org/tacl-submission-templates/"
TACL_STYLE_SHA256 = {
    "tacl2021v1.sty": "7c71ede112ccb3711c7073c4b1c60abfb076b83945e0e9d14ed79caa37dca3b6",
    "acl_natbib.bst": "27a3b62fdc4211fa56fe4827572ff0524b3ba3fe96c9443faaba5a683af83c08",
    "tacl2021v1-template.tex": "b2c11b9c996b07d733b52f1f9c2fb2332cafab57610465d072271bea504c6021",
}
PRIMARY_NAMES = (
    "delta_cer_P0_minus_qwen3asr",
    "delta_cer_P0_minus_S2",
    "delta_ece_has_minus_no",
    "delta_pp_pass_0.9",
    "delta_retention_P0_minus_S2",
    "logit_OR_has_dialect_P0",
)
CER_SYSTEMS = (
    ("sensevoice", "SenseVoice-Small", "CERsensevoice"),
    ("firered", "FireRedASR-AED-L", "CERfirered"),
    ("qwen3asr", "Qwen3-ASR-1.7B", "CERqwen"),
    ("vote_P0", "Vote P0", "CERvoteP"),
    ("vote_S2", "Vote S2", "CERvoteS"),
)
SPLITS = ("Easy", "Hard", "All")
CALIB_NAMES = (
    "ece3_vote_P0",
    "ece3_vote_S2",
    "spearman_conf_cer_vote_P0",
    "spearman_conf_cer_vote_S2",
)
PRIMARY_LABELS = {
    "delta_cer_P0_minus_qwen3asr": r"$\Delta$CER (P0 $-$ Qwen3-ASR)",
    "delta_cer_P0_minus_S2": r"$\Delta$CER (P0 $-$ S2)",
    "delta_ece_has_minus_no": r"$\Delta$ECE (dialect $-$ not)",
    "delta_pp_pass_0.9": r"$\Delta$pp pass $\geq$ threshold",
    "delta_retention_P0_minus_S2": r"$\Delta$retention (P0 $-$ S2)",
    "logit_OR_has_dialect_P0": r"OR (has dialect $\mid$ duration, split)",
}
SNAPSHOT_MSG = (
    "download both files from Drive `MyDrive/wsc-audit/outputs/metrics/` "
    "into `results/frozen/`, then re-run"
)
FROZEN_README = (
    "Byte copies of frozen analysis outputs; never edit; hashes in paper/PAPER_PROVENANCE.txt.\n"
)


class PaperInputError(RuntimeError):
    """Missing or inconsistent frozen paper inputs."""


def latex_escape(text: str) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("%", r"\%").replace("_", r"\_")


def write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\r\n", "\n").replace("\r", "\n")
    if not data.endswith("\n"):
        data += "\n"
    path.write_bytes(data.encode("utf-8"))


def fmt_cer(value: float | str, *, signed: bool = False) -> str:
    v = float(value)
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def fmt_sig(value: float | str) -> str:
    tr = Trace()
    return tr.sig(value, file="_", row_key="_", column="_")


def fmt_pct(value: float | str) -> str:
    return f"{100.0 * float(value):.1f}"


def fmt_ci(lo: float | str, hi: float | str, *, kind: str) -> str:
    if kind == "cer":
        return f"[{fmt_cer(lo)}, {fmt_cer(hi)}]"
    if kind == "cer_signed":
        return f"[{fmt_cer(lo, signed=True)}, {fmt_cer(hi, signed=True)}]"
    if kind == "pct":
        return f"[{fmt_pct(lo)}, {fmt_pct(hi)}]"
    return f"[{fmt_sig(lo)}, {fmt_sig(hi)}]"


def fmt_p(value: Any, *, B: int) -> str:
    if value is None or str(value).strip() == "":
        return "---"
    x = float(value)
    if x <= (1.0 / B) + 1e-18:
        return r"$<10^{-4}$"
    return f"{x:.3g}"


def parse_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise PaperInputError(f"missing file: {path} expected a readable file actual missing")
    return path


def _check_hash(path: Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise PaperInputError(
            f"hash mismatch for {label} file={path} expected {expected} actual {actual}"
        )
    return actual


def load_stats_table(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    _require_file(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r.get("name") or "", r.get("split") or "")
        if key in idx:
            raise PaperInputError(
                f"duplicate (name, split) file={path} expected unique pair actual {key}"
            )
        idx[key] = r
    return idx


def require_row(
    idx: dict[tuple[str, str], dict[str, str]], name: str, split: str, *, file: Path
) -> dict[str, str]:
    row = idx.get((name, split))
    if row is None:
        raise PaperInputError(
            f"missing required row file={file} expected name={name} split={split} actual missing"
        )
    return row


def load_summary(path: Path) -> dict[str, Any]:
    _require_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("n", "B", "seed"):
        if key not in data:
            raise PaperInputError(
                f"missing key in stats_summary.json file={path} expected {key} actual missing"
            )
    return data


def load_findings(path: Path) -> list[dict[str, Any]]:
    _require_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise PaperInputError(f"audit_findings_numbers.json file={path} expected list actual {type(data)}")
    return data


def findings_get(
    records: list[dict[str, Any]], row_key: str, column: str, *, file: str | None = None
) -> dict[str, Any]:
    hits = [
        r
        for r in records
        if r.get("row_key") == row_key and r.get("column") == column and (file is None or r.get("file") == file)
    ]
    if not hits:
        raise PaperInputError(
            f"missing findings entry expected row_key={row_key} column={column} actual missing"
        )
    return hits[0]


def findings_num(records: list[dict[str, Any]], row_key: str, column: str, *, file: str | None = None) -> float:
    raw = findings_get(records, row_key, column, file=file)["raw_value"]
    return float(raw)


def findings_int(records: list[dict[str, Any]], row_key: str, column: str, *, file: str | None = None) -> int:
    return int(float(findings_get(records, row_key, column, file=file)["raw_value"]))


def filter_thr_from_name(name: str) -> str:
    m = re.search(r"pass_([0-9.]+)$", name)
    if not m:
        raise PaperInputError(f"cannot parse filter threshold from row name expected pass_<thr> actual {name}")
    return m.group(1)


def or_cmp(value: float) -> str:
    return r"$<1$" if value < 1.0 else r"$>1$"


def resolve_stats_paths(
    *,
    repo_root: Path,
    stats_table: Path | None,
    stats_summary: Path | None,
    out_root: Path | None,
) -> tuple[Path, Path]:
    frozen_dir = repo_root / "results" / "frozen"
    frozen_table = frozen_dir / "stats_table.csv"
    frozen_sum = frozen_dir / "stats_summary.json"
    metrics = (out_root / "metrics") if out_root is not None else None
    env_table = (metrics / "stats_table.csv") if metrics is not None else None
    env_sum = (metrics / "stats_summary.json") if metrics is not None else None

    def pick(explicit: Path | None, frozen: Path, env: Path | None, name: str) -> Path:
        if explicit is not None:
            return _require_file(explicit)
        if frozen.is_file() and env is not None and env.is_file():
            h_a, h_b = sha256_file(frozen), sha256_file(env)
            if h_a != h_b:
                raise PaperInputError(
                    f"hash mismatch between copies of {name} expected {h_a} actual {h_b}"
                )
            return frozen
        if frozen.is_file():
            return frozen
        if env is not None and env.is_file():
            return env
        raise PaperInputError(f"missing file: {frozen} expected {name} actual missing")

    return (
        pick(stats_table, frozen_table, env_table, "stats_table.csv"),
        pick(stats_summary, frozen_sum, env_sum, "stats_summary.json"),
    )


def snapshot_frozen(*, repo_root: Path, out_root: Path | None) -> tuple[Path, Path]:
    frozen = repo_root / "results" / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    write_lf(frozen / "README.md", FROZEN_README)
    src_root = out_root / "metrics" if out_root is not None else None
    src_table = src_root / "stats_table.csv" if src_root is not None else None
    src_sum = src_root / "stats_summary.json" if src_root is not None else None
    if src_table is None or not src_table.is_file() or src_sum is None or not src_sum.is_file():
        print(SNAPSHOT_MSG)
        raise PaperInputError(SNAPSHOT_MSG)
    for src, dest in ((src_table, frozen / "stats_table.csv"), (src_sum, frozen / "stats_summary.json")):
        h_src = sha256_file(src)
        if dest.is_file():
            h_dest = sha256_file(dest)
            if h_dest != h_src:
                raise PaperInputError(
                    f"hash mismatch for snapshot target file={dest} expected {h_src} actual {h_dest}"
                )
        else:
            dest.write_bytes(src.read_bytes())
        if dest.name == "stats_table.csv":
            _check_hash(dest, STATS_TABLE_SHA256, label="stats_table.csv")
    return frozen / "stats_table.csv", frozen / "stats_summary.json"


class MacroBook:
    def __init__(self) -> None:
        self.macros: dict[str, str] = {}
        self.sources: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        formatted: str,
        *,
        file: str,
        row_key: str,
        column: str,
        raw_value: Any,
    ) -> str:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise PaperInputError(f"macro name must be letters only expected [A-Za-z]+ actual {name}")
        if name in self.macros:
            raise PaperInputError(f"duplicate macro expected unique actual {name}")
        self.macros[name] = formatted
        rec = {
            "macro_or_cell": name,
            "file": file,
            "row_key": row_key,
            "column": column,
            "raw_value": raw_value if isinstance(raw_value, (int, float, str, bool)) or raw_value is None else str(raw_value),
            "formatted": formatted,
        }
        self.sources.append(rec)
        return formatted

    def cell(
        self,
        cell_id: str,
        formatted: str,
        *,
        file: str,
        row_key: str,
        column: str,
        raw_value: Any,
    ) -> str:
        self.sources.append(
            {
                "macro_or_cell": cell_id,
                "file": file,
                "row_key": row_key,
                "column": column,
                "raw_value": raw_value if isinstance(raw_value, (int, float, str, bool)) or raw_value is None else str(raw_value),
                "formatted": formatted,
            }
        )
        return formatted


def _empty(v: Any) -> bool:
    return v is None or str(v).strip() == ""


def _float(row: dict[str, str], col: str) -> float:
    return float(row[col])


def render_tab_cer(
    idx: dict[tuple[str, str], dict[str, str]],
    book: MacroBook,
    *,
    stats_file: str,
    stats_path: Path,
) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"System & Easy & Hard & All \\",
        r"\midrule",
    ]
    for stem, label, _macro in CER_SYSTEMS:
        cells = []
        for split in SPLITS:
            row = require_row(idx, f"cer_{stem}", split, file=stats_path)
            est = fmt_cer(row["estimate"])
            ci = fmt_ci(row["ci_lo"], row["ci_hi"], kind="cer")
            cell = f"{est} {ci}"
            book.cell(
                f"tab_cer:{stem}:{split}",
                cell,
                file=stats_file,
                row_key=f"cer_{stem}",
                column=f"{split}:estimate_ci",
                raw_value=row["estimate"],
            )
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_tab_primary(
    idx: dict[tuple[str, str], dict[str, str]],
    book: MacroBook,
    *,
    stats_file: str,
    stats_path: Path,
    B: int,
) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Contrast & Estimate & 95\% CI (percentile bootstrap; Wald for the OR) & $p$ (raw) & $p$ (Holm) \\",
        r"\midrule",
    ]
    for name in PRIMARY_NAMES:
        row = require_row(idx, name, "All", file=stats_path)
        if name.startswith("delta_cer"):
            est = fmt_cer(row["estimate"], signed=True)
            ci = fmt_ci(row["ci_lo"], row["ci_hi"], kind="cer_signed")
        else:
            est = fmt_sig(row["estimate"])
            ci = fmt_ci(row["ci_lo"], row["ci_hi"], kind="sig")
        p_raw = fmt_p(row.get("p_raw"), B=B)
        p_holm = fmt_p(row.get("p_holm"), B=B)
        book.cell(
            f"tab_primary:{name}:estimate",
            est,
            file=stats_file,
            row_key=name,
            column="estimate",
            raw_value=row["estimate"],
        )
        book.cell(
            f"tab_primary:{name}:ci",
            ci,
            file=stats_file,
            row_key=name,
            column="ci",
            raw_value=row["ci_lo"],
        )
        label = PRIMARY_LABELS[name]
        lines.append(f"{label} & {est} & {ci} & {p_raw} & {p_holm} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}{\small $\Delta$pp $p$ is a within-split permutation $p$.}",
    ]
    return "\n".join(lines) + "\n"


def render_tab_calib(
    idx: dict[tuple[str, str], dict[str, str]],
    book: MacroBook,
    *,
    stats_file: str,
    stats_path: Path,
) -> str:
    labels = {
        "ece3_vote_P0": r"ECE$_3$ Vote P0",
        "ece3_vote_S2": r"ECE$_3$ Vote S2",
        "spearman_conf_cer_vote_P0": r"Spearman $\rho$ (conf, CER) P0",
        "spearman_conf_cer_vote_S2": r"Spearman $\rho$ (conf, CER) S2",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Easy & Hard & All \\",
        r"\midrule",
    ]
    for name in CALIB_NAMES:
        cells = []
        for split in SPLITS:
            row = require_row(idx, name, split, file=stats_path)
            est = fmt_sig(row["estimate"])
            ci = fmt_ci(row["ci_lo"], row["ci_hi"], kind="sig")
            cell = f"{est} {ci}"
            book.cell(
                f"tab_calib:{name}:{split}",
                cell,
                file=stats_file,
                row_key=name,
                column=f"{split}:estimate_ci",
                raw_value=row["estimate"],
            )
            cells.append(cell)
        lines.append(f"{labels[name]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_tab_audit(
    findings: list[dict[str, Any]],
    book: MacroBook,
    *,
    audit_table_tex: str,
    findings_file: str,
) -> str:
    mand_n = findings_int(findings, "precision_mandarinization", "n")
    mand_k = findings_int(findings, "precision_mandarinization", "k")
    mand_p = fmt_pct(findings_num(findings, "precision_mandarinization", "estimate"))
    mand_ci = fmt_ci(
        findings_num(findings, "precision_mandarinization", "ci_low"),
        findings_num(findings, "precision_mandarinization", "ci_high"),
        kind="pct",
    )
    mand_prec = f"{mand_p}\\% {mand_ci}"
    rec_p = fmt_pct(findings_num(findings, "recall_mandarinization_weighted", "estimate"))
    rec_ci = fmt_ci(
        findings_num(findings, "recall_mandarinization_weighted", "ci_low"),
        findings_num(findings, "recall_mandarinization_weighted", "ci_high"),
        kind="pct",
    )
    mand_rec = f"{rec_p}\\% {rec_ci}"
    over_n = findings_int(findings, "precision_over_dialect", "n")
    over_k = findings_int(findings, "precision_over_dialect", "k")
    over_p = fmt_pct(findings_num(findings, "precision_over_dialect", "estimate"))
    over_ci = fmt_ci(
        findings_num(findings, "precision_over_dialect", "ci_low"),
        findings_num(findings, "precision_over_dialect", "ci_high"),
        kind="pct",
    )
    over_prec = f"{over_p}\\% {over_ci}"
    low_k = findings_int(findings, "control_lowconf_error_rate", "k")
    low_n = findings_int(findings, "control_lowconf_error_rate", "n")
    low_p = fmt_pct(findings_num(findings, "control_lowconf_error_rate", "estimate"))
    low_ci = fmt_ci(
        findings_num(findings, "control_lowconf_error_rate", "ci_low"),
        findings_num(findings, "control_lowconf_error_rate", "ci_high"),
        kind="pct",
    )
    low_rate = f"{low_p}\\% {low_ci}"
    for displayed in (mand_prec, mand_rec, over_prec):
        if displayed not in audit_table_tex:
            raise PaperInputError(
                f"tab_audit string missing from audit_table.tex expected {displayed!r} actual not found"
            )
    book.cell("tab_audit:mand:prec", mand_prec, file=findings_file, row_key="precision_mandarinization", column="estimate", raw_value=mand_p)
    book.cell("tab_audit:mand:rec", mand_rec, file=findings_file, row_key="recall_mandarinization_weighted", column="estimate", raw_value=rec_p)
    book.cell("tab_audit:over:prec", over_prec, file=findings_file, row_key="precision_over_dialect", column="estimate", raw_value=over_p)
    book.cell("tab_audit:low:rate", low_rate, file=findings_file, row_key="control_lowconf_error_rate", column="estimate", raw_value=low_p)
    lines = [
        r"\begin{tabular}{lrrll}",
        r"\toprule",
        r"Detector & $n$ flagged & Human-confirmed & Precision [Wilson 95\% CI] & Recall (weighted) [bootstrap 95\% CI] \\",
        r"\midrule",
        f"Mandarinization & {mand_n} & {mand_k} & {mand_prec} & {mand_rec} \\\\",
        f"Over-dialect & {over_n} & {over_k} & {over_prec} & --- \\\\",
        f"Low-confidence controls (any error) & {low_n} & {low_k} & {low_rate} & --- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{2pt}{\small The over-dialect detector is unreliable; its automatic counts are not a result.}",
    ]
    return "\n".join(lines) + "\n"


def fill_macros(
    book: MacroBook,
    *,
    idx: dict[tuple[str, str], dict[str, str]],
    summary: dict[str, Any],
    findings: list[dict[str, Any]],
    stats_file: str,
    stats_path: Path,
    B: int,
) -> None:
    book.add("NUtt", str(int(summary["n"])), file="stats_summary.json", row_key="n", column="n", raw_value=summary["n"])
    book.add("NBoot", str(int(summary["B"])), file="stats_summary.json", row_key="B", column="B", raw_value=summary["B"])
    book.add("BootSeed", str(int(summary["seed"])), file="stats_summary.json", row_key="seed", column="seed", raw_value=summary["seed"])

    macro_map = {stem: macro for stem, _label, macro in CER_SYSTEMS}
    for stem, _label, macro in CER_SYSTEMS:
        row = require_row(idx, f"cer_{stem}", "All", file=stats_path)
        book.add(macro, fmt_cer(row["estimate"]), file=stats_file, row_key=f"cer_{stem}", column="estimate", raw_value=row["estimate"])
        book.add(
            f"{macro}CI",
            fmt_ci(row["ci_lo"], row["ci_hi"], kind="cer"),
            file=stats_file,
            row_key=f"cer_{stem}",
            column="ci",
            raw_value=row["ci_lo"],
        )

    def add_delta_cer(name: str, est_m: str, ci_m: str, p_m: str) -> None:
        row = require_row(idx, name, "All", file=stats_path)
        book.add(est_m, fmt_cer(row["estimate"], signed=True), file=stats_file, row_key=name, column="estimate", raw_value=row["estimate"])
        book.add(ci_m, fmt_ci(row["ci_lo"], row["ci_hi"], kind="cer_signed"), file=stats_file, row_key=name, column="ci", raw_value=row["ci_lo"])
        book.add(p_m, fmt_p(row.get("p_holm"), B=B), file=stats_file, row_key=name, column="p_holm", raw_value=row.get("p_holm"))

    add_delta_cer("delta_cer_P0_minus_qwen3asr", "DeltaCERPqwen", "DeltaCERPqwenCI", "DeltaCERPqwenpHolm")
    add_delta_cer("delta_cer_P0_minus_S2", "DeltaCERPS", "DeltaCERPSCI", "DeltaCERPSpHolm")

    ece = require_row(idx, "delta_ece_has_minus_no", "All", file=stats_path)
    book.add("DeltaECE", fmt_sig(ece["estimate"]), file=stats_file, row_key="delta_ece_has_minus_no", column="estimate", raw_value=ece["estimate"])
    book.add("DeltaECECI", fmt_ci(ece["ci_lo"], ece["ci_hi"], kind="sig"), file=stats_file, row_key="delta_ece_has_minus_no", column="ci", raw_value=ece["ci_lo"])
    book.add("DeltaECEpHolm", fmt_p(ece.get("p_holm"), B=B), file=stats_file, row_key="delta_ece_has_minus_no", column="p_holm", raw_value=ece.get("p_holm"))

    pp = require_row(idx, "delta_pp_pass_0.9", "All", file=stats_path)
    thr = filter_thr_from_name("delta_pp_pass_0.9")
    book.add("FilterThr", thr, file=stats_file, row_key="delta_pp_pass_0.9", column="name", raw_value=thr)
    book.add("DeltaPP", fmt_sig(pp["estimate"]), file=stats_file, row_key="delta_pp_pass_0.9", column="estimate", raw_value=pp["estimate"])
    book.add("DeltaPPCI", fmt_ci(pp["ci_lo"], pp["ci_hi"], kind="sig"), file=stats_file, row_key="delta_pp_pass_0.9", column="ci", raw_value=pp["ci_lo"])
    book.add("DeltaPPpHolm", fmt_p(pp.get("p_holm"), B=B), file=stats_file, row_key="delta_pp_pass_0.9", column="p_holm", raw_value=pp.get("p_holm"))
    pp_est = float(pp["estimate"])
    book.add("FilterPassWord", delta_pass_word(pp_est), file=stats_file, row_key="delta_pp_pass_0.9", column="sign_word", raw_value=pp_est)

    lor = require_row(idx, "logit_OR_has_dialect_P0", "All", file=stats_path)
    or_v = float(lor["estimate"])
    book.add("ORdialect", fmt_sig(or_v), file=stats_file, row_key="logit_OR_has_dialect_P0", column="estimate", raw_value=lor["estimate"])
    book.add("ORdialectCI", fmt_ci(lor["ci_lo"], lor["ci_hi"], kind="sig"), file=stats_file, row_key="logit_OR_has_dialect_P0", column="ci", raw_value=lor["ci_lo"])
    book.add("ORpHolm", fmt_p(lor.get("p_holm"), B=B), file=stats_file, row_key="logit_OR_has_dialect_P0", column="p_holm", raw_value=lor.get("p_holm"))
    book.add("ORcmp", or_cmp(or_v), file=stats_file, row_key="logit_OR_has_dialect_P0", column="cmp", raw_value=or_v)

    p_pp = float(pp["p_holm"])
    p_or = float(lor["p_holm"])
    book.add("HolmClause", holm_clause(p_pp, p_or), file=stats_file, row_key="p_holm", column="clause", raw_value=f"{p_pp},{p_or}")

    book.add("AuditN", str(findings_int(findings, "n_rows", "n")), file="audit_findings_numbers.json", row_key="n_rows", column="n", raw_value=findings_int(findings, "n_rows", "n"))
    book.add("AuditNCtrlDialect", str(findings_int(findings, "auto_category:control_dialect", "n")), file="audit_findings_numbers.json", row_key="auto_category:control_dialect", column="n", raw_value=findings_int(findings, "auto_category:control_dialect", "n"))
    book.add("AuditNCtrlLowconf", str(findings_int(findings, "auto_category:control_lowconf", "n")), file="audit_findings_numbers.json", row_key="auto_category:control_lowconf", column="n", raw_value=findings_int(findings, "auto_category:control_lowconf", "n"))
    book.add("AuditNMand", str(findings_int(findings, "auto_category:mandarinization", "n")), file="audit_findings_numbers.json", row_key="auto_category:mandarinization", column="n", raw_value=findings_int(findings, "auto_category:mandarinization", "n"))
    book.add("AuditNOver", str(findings_int(findings, "auto_category:over_dialect", "n")), file="audit_findings_numbers.json", row_key="auto_category:over_dialect", column="n", raw_value=findings_int(findings, "auto_category:over_dialect", "n"))
    book.add("NRefErrorOnly", str(findings_int(findings, "n_ref_error_only", "k")), file="audit_findings_numbers.json", row_key="n_ref_error_only", column="k", raw_value=findings_int(findings, "n_ref_error_only", "k"))

    book.add("MandPrecK", str(findings_int(findings, "precision_mandarinization", "k")), file="audit_findings_numbers.json", row_key="precision_mandarinization", column="k", raw_value=findings_int(findings, "precision_mandarinization", "k"))
    book.add("MandPrecN", str(findings_int(findings, "precision_mandarinization", "n")), file="audit_findings_numbers.json", row_key="precision_mandarinization", column="n", raw_value=findings_int(findings, "precision_mandarinization", "n"))
    book.add("MandPrec", fmt_pct(findings_num(findings, "precision_mandarinization", "estimate")), file="audit_findings_numbers.json", row_key="precision_mandarinization", column="estimate", raw_value=findings_num(findings, "precision_mandarinization", "estimate"))
    book.add(
        "MandPrecCI",
        fmt_ci(
            findings_num(findings, "precision_mandarinization", "ci_low"),
            findings_num(findings, "precision_mandarinization", "ci_high"),
            kind="pct",
        ),
        file="audit_findings_numbers.json",
        row_key="precision_mandarinization",
        column="ci",
        raw_value=findings_num(findings, "precision_mandarinization", "ci_low"),
    )
    book.add("MandRecallW", fmt_pct(findings_num(findings, "recall_mandarinization_weighted", "estimate")), file="audit_findings_numbers.json", row_key="recall_mandarinization_weighted", column="estimate", raw_value=findings_num(findings, "recall_mandarinization_weighted", "estimate"))
    book.add(
        "MandRecallWCI",
        fmt_ci(
            findings_num(findings, "recall_mandarinization_weighted", "ci_low"),
            findings_num(findings, "recall_mandarinization_weighted", "ci_high"),
            kind="pct",
        ),
        file="audit_findings_numbers.json",
        row_key="recall_mandarinization_weighted",
        column="ci",
        raw_value=findings_num(findings, "recall_mandarinization_weighted", "ci_low"),
    )
    book.add("NCtrlPop", str(findings_int(findings, "recall_mandarinization_weighted", "note:N_ctrl_pop")), file="audit_findings_numbers.json", row_key="recall_mandarinization_weighted", column="note:N_ctrl_pop", raw_value=findings_int(findings, "recall_mandarinization_weighted", "note:N_ctrl_pop"))
    book.add("MissesCtrl", str(findings_int(findings, "recall_mandarinization_weighted", "note:misses_ctrl")), file="audit_findings_numbers.json", row_key="recall_mandarinization_weighted", column="note:misses_ctrl", raw_value=findings_int(findings, "recall_mandarinization_weighted", "note:misses_ctrl"))
    book.add("NCtrlDesign", str(findings_int(findings, "recall_mandarinization_weighted", "note:design_n")), file="audit_findings_numbers.json", row_key="recall_mandarinization_weighted", column="note:design_n", raw_value=findings_int(findings, "recall_mandarinization_weighted", "note:design_n"))

    book.add("OverPrecK", str(findings_int(findings, "precision_over_dialect", "k")), file="audit_findings_numbers.json", row_key="precision_over_dialect", column="k", raw_value=findings_int(findings, "precision_over_dialect", "k"))
    book.add("OverPrecN", str(findings_int(findings, "precision_over_dialect", "n")), file="audit_findings_numbers.json", row_key="precision_over_dialect", column="n", raw_value=findings_int(findings, "precision_over_dialect", "n"))
    book.add("OverPrec", fmt_pct(findings_num(findings, "precision_over_dialect", "estimate")), file="audit_findings_numbers.json", row_key="precision_over_dialect", column="estimate", raw_value=findings_num(findings, "precision_over_dialect", "estimate"))
    book.add(
        "OverPrecCI",
        fmt_ci(
            findings_num(findings, "precision_over_dialect", "ci_low"),
            findings_num(findings, "precision_over_dialect", "ci_high"),
            kind="pct",
        ),
        file="audit_findings_numbers.json",
        row_key="precision_over_dialect",
        column="ci",
        raw_value=findings_num(findings, "precision_over_dialect", "ci_low"),
    )

    book.add("LowconfK", str(findings_int(findings, "control_lowconf_error_rate", "k")), file="audit_findings_numbers.json", row_key="control_lowconf_error_rate", column="k", raw_value=findings_int(findings, "control_lowconf_error_rate", "k"))
    book.add("LowconfN", str(findings_int(findings, "control_lowconf_error_rate", "n")), file="audit_findings_numbers.json", row_key="control_lowconf_error_rate", column="n", raw_value=findings_int(findings, "control_lowconf_error_rate", "n"))
    book.add("LowconfRate", fmt_pct(findings_num(findings, "control_lowconf_error_rate", "estimate")), file="audit_findings_numbers.json", row_key="control_lowconf_error_rate", column="estimate", raw_value=findings_num(findings, "control_lowconf_error_rate", "estimate"))
    book.add(
        "LowconfCI",
        fmt_ci(
            findings_num(findings, "control_lowconf_error_rate", "ci_low"),
            findings_num(findings, "control_lowconf_error_rate", "ci_high"),
            kind="pct",
        ),
        file="audit_findings_numbers.json",
        row_key="control_lowconf_error_rate",
        column="ci",
        raw_value=findings_num(findings, "control_lowconf_error_rate", "ci_low"),
    )
    _ = macro_map


def render_numbers_tex(book: MacroBook) -> str:
    lines = [r"% GENERATED by wscaudit paper-tables — do not edit"]
    for name in book.macros:
        lines.append(f"\\newcommand{{\\{name}}}{{{book.macros[name]}\\xspace}}")
    return "\n".join(lines) + "\n"


def copy_figures(*, results_dir: Path, paper_dir: Path, out_root: Path | None, provenance: dict[str, str]) -> dict[str, str]:
    dest = paper_dir / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    src_pdf = results_dir / "figures" / "audit_precision_recall.pdf"
    _require_file(src_pdf)
    expected = provenance.get("output_sha256_audit_precision_recall.pdf")
    if expected:
        _check_hash(src_pdf, expected, label="audit_precision_recall.pdf")
    target = dest / "audit_precision_recall.pdf"
    target.write_bytes(src_pdf.read_bytes())
    if sha256_file(target) != sha256_file(src_pdf):
        raise PaperInputError(f"figure copy hash mismatch file={target}")
    hashes["audit_precision_recall.pdf"] = sha256_file(target)
    src_png = results_dir / "figures" / "audit_precision_recall.png"
    if src_png.is_file():
        (dest / src_png.name).write_bytes(src_png.read_bytes())
        hashes[src_png.name] = sha256_file(dest / src_png.name)
    fig_dirs = []
    if out_root is not None:
        fig_dirs.append(out_root / "metrics" / "figs")
    fig_dirs.append(results_dir / "metrics" / "figs")
    for d in fig_dirs:
        if not d.is_dir():
            continue
        for name in ("reliability_All.png", "reliability_Easy.png", "reliability_Hard.png"):
            p = d / name
            if p.is_file():
                (dest / name).write_bytes(p.read_bytes())
                hashes[name] = sha256_file(dest / name)
    return hashes


def tacl_style_hashes(paper_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("tacl2021v1.sty", "acl_natbib.bst"):
        p = paper_dir / name
        if p.is_file():
            out[name] = sha256_file(p)
            expected = TACL_STYLE_SHA256.get(name)
            if expected and out[name] != expected:
                print(f"warning: {name} sha256 {out[name]} does not match kit hash {expected}")
        kit = paper_dir / "_kit" / "tacl2021v1-template.tex"
        if name == "tacl2021v1.sty" and kit.is_file():
            h = sha256_file(kit)
            out["tacl2021v1-template.tex"] = h
            exp = TACL_STYLE_SHA256["tacl2021v1-template.tex"]
            if h != exp:
                print(f"warning: tacl2021v1-template.tex sha256 {h} does not match kit hash {exp}")
    return out


def write_provenance(
    path: Path,
    *,
    repo_root: Path,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    stats_before: str,
    stats_after: str,
    style_hashes: dict[str, str],
) -> None:
    if stats_before != stats_after:
        raise PaperInputError(
            f"stats_table sha256 before/after differ expected {stats_before} actual {stats_after}"
        )
    lines = [f"git_commit={git_head(repo_root)}"]
    for name in sorted(inputs):
        lines.append(f"input_sha256_{name}={sha256_file(inputs[name])}")
    for name in sorted(outputs):
        p = outputs[name]
        if p.is_file():
            lines.append(f"output_sha256_{name}={sha256_file(p)}")
    lines.append(f"stats_table_sha256_before={stats_before}")
    lines.append(f"stats_table_sha256_after={stats_after}")
    for name, h in sorted(style_hashes.items()):
        lines.append(f"tacl_style_sha256_{name}={h}")
    lines.append(f"tacl_style_source={TACL_STYLE_SOURCE}")
    lines.append("naive unweighted recall: not reported")
    lines.append("cpu_only=true")
    write_lf(path, "\n".join(lines) + "\n")


def _dir_diff(expected: Path, actual: Path) -> list[str]:
    diffs: list[str] = []
    names = sorted({p.name for p in expected.rglob("*") if p.is_file()} | {p.name for p in actual.rglob("*") if p.is_file()})
    exp_map = {p.relative_to(expected).as_posix(): p for p in expected.rglob("*") if p.is_file()}
    act_map = {p.relative_to(actual).as_posix(): p for p in actual.rglob("*") if p.is_file()}
    for rel in sorted(set(exp_map) | set(act_map)):
        if rel.endswith("PAPER_PROVENANCE.txt"):
            # git_commit can differ only if regenerated; still compare body hashes except git_commit line
            pass
        a, b = exp_map.get(rel), act_map.get(rel)
        if a is None or b is None:
            diffs.append(rel)
            continue
        if a.read_bytes() != b.read_bytes():
            if rel.endswith("PAPER_PROVENANCE.txt"):
                def drop_commit(data: bytes) -> bytes:
                    lines = data.decode("utf-8").splitlines()
                    return "\n".join(x for x in lines if not x.startswith("git_commit=")).encode()

                if drop_commit(a.read_bytes()) == drop_commit(b.read_bytes()):
                    continue
            diffs.append(rel)
    return diffs


GENERATED_REL = (
    "numbers.tex",
    "numbers_sources.json",
    "PAPER_PROVENANCE.txt",
    "tables/tab_cer.tex",
    "tables/tab_primary.tex",
    "tables/tab_calib.tex",
    "tables/tab_audit.tex",
    "figures/audit_precision_recall.pdf",
)


def run_paper_tables(
    *,
    repo_root: Path | None = None,
    stats_table: Path | None = None,
    stats_summary: Path | None = None,
    results_dir: Path | None = None,
    paper_dir: Path | None = None,
    out_root: Path | None = None,
    snapshot: bool = False,
    check: bool = False,
    expected_stats_sha256: str | None = None,
) -> dict[str, Path]:
    repo_root = repo_root or _REPO_ROOT
    results_dir = results_dir or (repo_root / "results")
    paper_dir = paper_dir or (repo_root / "paper")
    expected_hash = expected_stats_sha256 or STATS_TABLE_SHA256

    if snapshot:
        stats_table, stats_summary = snapshot_frozen(repo_root=repo_root, out_root=out_root)
    else:
        stats_table, stats_summary = resolve_stats_paths(
            repo_root=repo_root,
            stats_table=stats_table,
            stats_summary=stats_summary,
            out_root=out_root,
        )

    frozen_readme = repo_root / "results" / "frozen" / "README.md"
    if stats_table.parent.name == "frozen" and not frozen_readme.is_file():
        write_lf(frozen_readme, FROZEN_README)

    work_dir = paper_dir
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if check:
        tmp = tempfile.TemporaryDirectory()
        work_dir = Path(tmp.name) / "paper"
        work_dir.mkdir(parents=True)

    try:
        return _generate(
            repo_root=repo_root,
            stats_table=stats_table,
            stats_summary=stats_summary,
            results_dir=results_dir,
            paper_dir=work_dir,
            out_root=out_root,
            expected_hash=expected_hash,
            check_against=paper_dir if check else None,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def _generate(
    *,
    repo_root: Path,
    stats_table: Path,
    stats_summary: Path,
    results_dir: Path,
    paper_dir: Path,
    out_root: Path | None,
    expected_hash: str,
    check_against: Path | None,
) -> dict[str, Path]:
    prov_audit = parse_kv_file(_require_file(results_dir / "AUDIT_REPORT_PROVENANCE.txt"))
    h_table = _check_hash(stats_table, expected_hash, label="stats_table.csv")
    if expected_hash == STATS_TABLE_SHA256:
        after = prov_audit.get("stats_table_sha256_after", h_table)
        if after != h_table:
            raise PaperInputError(
                f"hash mismatch vs AUDIT_REPORT_PROVENANCE.txt file={stats_table} expected {after} actual {h_table}"
            )
    findings_path = _require_file(results_dir / "audit_findings_numbers.json")
    exp_findings = prov_audit.get("output_sha256_audit_findings_numbers.json")
    if exp_findings:
        _check_hash(findings_path, exp_findings, label="audit_findings_numbers.json")
    tex_path = _require_file(results_dir / "audit_table.tex")
    exp_tex = prov_audit.get("input_sha256_audit_table.tex")
    if exp_tex:
        _check_hash(tex_path, exp_tex, label="audit_table.tex")

    idx = load_stats_table(stats_table)
    for name in PRIMARY_NAMES:
        require_row(idx, name, "All", file=stats_table)
    summary = load_summary(stats_summary)
    B = int(summary["B"])
    findings = load_findings(findings_path)
    audit_tex = tex_path.read_text(encoding="utf-8")
    forbidden = ("23/45", "naive", "unweighted")
    # generator must not emit these into paper outputs; checked after write

    book = MacroBook()
    stats_file = stats_table.name
    tab_cer = render_tab_cer(idx, book, stats_file=stats_file, stats_path=stats_table)
    fill_macros(book, idx=idx, summary=summary, findings=findings, stats_file=stats_file, stats_path=stats_table, B=B)
    tab_primary = render_tab_primary(idx, book, stats_file=stats_file, stats_path=stats_table, B=B)
    tab_calib = render_tab_calib(idx, book, stats_file=stats_file, stats_path=stats_table)
    tab_audit = render_tab_audit(findings, book, audit_table_tex=audit_tex, findings_file=findings_path.name)

    tables = paper_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {
        "tab_cer.tex": tables / "tab_cer.tex",
        "tab_primary.tex": tables / "tab_primary.tex",
        "tab_calib.tex": tables / "tab_calib.tex",
        "tab_audit.tex": tables / "tab_audit.tex",
        "numbers.tex": paper_dir / "numbers.tex",
        "numbers_sources.json": paper_dir / "numbers_sources.json",
        "PAPER_PROVENANCE.txt": paper_dir / "PAPER_PROVENANCE.txt",
    }
    write_lf(paths["tab_cer.tex"], tab_cer)
    write_lf(paths["tab_primary.tex"], tab_primary)
    write_lf(paths["tab_calib.tex"], tab_calib)
    write_lf(paths["tab_audit.tex"], tab_audit)
    write_lf(paths["numbers.tex"], render_numbers_tex(book))
    sidecar = sorted(book.sources, key=lambda r: (r["macro_or_cell"], r["row_key"], r["column"]))
    write_lf(paths["numbers_sources.json"], json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    fig_hashes = copy_figures(results_dir=results_dir, paper_dir=paper_dir, out_root=out_root, provenance=prov_audit)
    style_hashes = tacl_style_hashes(paper_dir if check_against is None else check_against)

    inputs = {
        "stats_table.csv": stats_table,
        "stats_summary.json": stats_summary,
        "audit_findings_numbers.json": findings_path,
        "audit_table.tex": tex_path,
        "AUDIT_REPORT_PROVENANCE.txt": results_dir / "AUDIT_REPORT_PROVENANCE.txt",
        "audit_precision_recall.pdf": results_dir / "figures" / "audit_precision_recall.pdf",
    }
    outputs = {
        "numbers.tex": paths["numbers.tex"],
        "numbers_sources.json": paths["numbers_sources.json"],
        "tab_cer.tex": paths["tab_cer.tex"],
        "tab_primary.tex": paths["tab_primary.tex"],
        "tab_calib.tex": paths["tab_calib.tex"],
        "tab_audit.tex": paths["tab_audit.tex"],
        "audit_precision_recall.pdf": paper_dir / "figures" / "audit_precision_recall.pdf",
    }
    write_provenance(
        paths["PAPER_PROVENANCE.txt"],
        repo_root=repo_root,
        inputs=inputs,
        outputs=outputs,
        stats_before=h_table,
        stats_after=h_table,
        style_hashes=style_hashes,
    )

    blob = "\n".join(
        p.read_text(encoding="utf-8")
        for k, p in paths.items()
        if k != "PAPER_PROVENANCE.txt" and p.suffix in {".tex", ".json", ".md"}
    )
    for bad in ("23/45", "less often", "bias is proven", "naive", "unweighted"):
        if bad in blob:
            raise PaperInputError(f"forbidden phrase {bad!r} in generated paper files")

    _ = fig_hashes
    if check_against is not None:
        diffs = _dir_diff(check_against, paper_dir)
        # only generated artefacts
        gen_diffs = [d for d in diffs if any(d.endswith(x) or d == x for x in GENERATED_REL)]
        if gen_diffs:
            raise PaperInputError(f"--check differs: {gen_diffs}")
    return paths


def cmd_paper_tables(args: Any, cfg: dict[str, Any]) -> int:
    out_root = Path(cfg["out_root"]) if cfg.get("out_root") else None
    try:
        run_paper_tables(
            stats_table=args.stats_table,
            stats_summary=args.stats_summary,
            results_dir=args.results_dir,
            paper_dir=args.paper_dir,
            out_root=out_root,
            snapshot=args.snapshot,
            check=args.check,
        )
    except PaperInputError as exc:
        print(str(exc), flush=True)
        return 2
    return 0
