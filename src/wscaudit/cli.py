"""CLI: ``python -m wscaudit <step> ...``

Subcommands are stubs until later prompts fill them in. Paths come from
``configs/paths.yaml`` and env vars ``WSC_DATA``, ``WSC_OUT``, ``HF_HOME``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

SUBCOMMANDS = (
    "prepare",
    "asr",
    "merge",
    "vote",
    "confidence",
    "cer",
    "markers",
    "calibration",
    "stats",
    "report",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_paths(config_path: Path | None = None) -> dict[str, Any]:
    """Load path defaults and overlay env vars."""
    path = config_path or (_REPO_ROOT / "configs" / "paths.yaml")
    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    if os.environ.get("WSC_DATA"):
        cfg["data_root"] = os.environ["WSC_DATA"]
    if os.environ.get("WSC_OUT"):
        cfg["out_root"] = os.environ["WSC_OUT"]
    if os.environ.get("HF_HOME"):
        cfg["hf_home"] = os.environ["HF_HOME"]
    return cfg


def _stub(name: str) -> int:
    print(f"wscaudit {name}: not implemented yet", file=sys.stderr)
    return 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "paths.yaml",
        help="paths.yaml (env vars still win)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wscaudit",
        description="Re-run and audit the WenetSpeech-Chuan labelling pipeline on WSC-Eval-ASR.",
    )
    parser.add_argument("--version", action="version", version="wscaudit 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Download WSC-Eval-ASR and write manifest.jsonl")
    _add_common(p_prepare)

    p_asr = sub.add_parser("asr", help="Run one recogniser over the manifest")
    _add_common(p_asr)
    p_asr.add_argument("--model", required=False, help="firered | sensevoice | qwen3asr | whisper")
    p_asr.add_argument("--split", default="all", help="Easy | Hard | all")
    p_asr.add_argument("--device", default="auto")
    p_asr.add_argument("--limit", type=int, default=None)

    p_merge = sub.add_parser("merge", help="LLM-GER merge for a pipeline")
    _add_common(p_merge)
    p_merge.add_argument("--pipeline", default="P0")
    p_merge.add_argument("--device", default="auto")

    p_vote = sub.add_parser("vote", help="ROVER-style majority vote (no LLM)")
    _add_common(p_vote)
    p_vote.add_argument("--pipeline", default="P0-vote")

    p_conf = sub.add_parser("confidence", help="Position-wise confidence variants A/B")
    _add_common(p_conf)
    p_conf.add_argument("--pipeline", default="P0")

    p_cer = sub.add_parser("cer", help="Character error rates")
    _add_common(p_cer)

    p_markers = sub.add_parser("markers", help="Dialect marker loss/insertion")
    _add_common(p_markers)

    p_cal = sub.add_parser("calibration", help="ECE, reliability diagrams, filtering bias")
    _add_common(p_cal)

    p_stats = sub.add_parser("stats", help="Bootstrap, permutation tests, logistic regression")
    _add_common(p_stats)

    p_report = sub.add_parser("report", help="Comparison tables and provenance")
    _add_common(p_report)
    p_report.add_argument("--compare", nargs="*", default=None)
    p_report.add_argument("--env", action="store_true", help="Write results/provenance.json")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ = load_paths(args.config)
    return _stub(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
