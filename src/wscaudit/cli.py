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
    p_prepare.add_argument(
        "--skip-download",
        action="store_true",
        help="Use files already under data_root (no Hugging Face snapshot)",
    )

    p_asr = sub.add_parser("asr", help="Run one recogniser over the manifest")
    _add_common(p_asr)
    p_asr.add_argument("--model", required=True, help="firered | sensevoice | qwen3asr | whisper")
    p_asr.add_argument("--split", default="all", help="Easy | Hard | all")
    p_asr.add_argument("--device", default="auto")
    p_asr.add_argument("--limit", type=int, default=None)
    p_asr.add_argument("--batch-size", type=int, default=None)
    p_asr.add_argument(
        "--backend",
        default=None,
        help="qwen3asr only: transformers (default) or vllm",
    )

    p_merge = sub.add_parser("merge", help="LLM-GER merge for a pipeline")
    _add_common(p_merge)
    p_merge.add_argument("--pipeline", default="P0")
    p_merge.add_argument("--device", default="auto")

    p_vote = sub.add_parser("vote", help="Character-level majority vote (no LLM)")
    _add_common(p_vote)
    p_vote.add_argument("--pipeline", default="P0")
    p_vote.add_argument(
        "--hyps",
        default="sensevoice,firered,qwen3asr",
        help="Comma-separated hyp JSONL stems under out_root/hyps/",
    )
    p_vote.add_argument("--split", default="all", help="Easy | Hard | all")
    p_vote.add_argument("--limit", type=int, default=None)

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


def _apply_hf_home(cfg: dict[str, Any]) -> None:
    hf_home = cfg.get("hf_home")
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)


def load_models_cfg() -> dict[str, Any]:
    path = _REPO_ROOT / "configs" / "models.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cmd_prepare(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    from wscaudit.data import prepare

    _apply_hf_home(cfg)
    prepare(
        Path(cfg["data_root"]),
        Path(cfg["out_root"]),
        skip_download=args.skip_download,
    )
    return 0


def cmd_asr(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    from wscaudit.asr.base import run_asr

    _apply_hf_home(cfg)
    if args.model not in {"sensevoice", "firered", "qwen3asr"}:
        return _stub(f"asr --model {args.model}")
    models = load_models_cfg()
    asr_cfg = models.get("asr") or {}
    block = dict(asr_cfg.get(args.model) or {})
    if args.model == "qwen3asr" and args.backend:
        block["backend"] = args.backend
        asr_cfg = {**asr_cfg, "qwen3asr": block}
    default_batch = 8 if args.model in {"firered", "qwen3asr"} else 64
    batch = (
        args.batch_size
        if args.batch_size is not None
        else int(block.get("batch_size") or default_batch)
    )
    out_root = Path(cfg["out_root"])
    run_asr(
        args.model,
        out_root / "manifest.jsonl",
        out_root / "hyps" / f"{args.model}.jsonl",
        Path(cfg["data_root"]),
        batch_size=batch,
        device=args.device,
        split=args.split,
        limit=args.limit,
        asr_cfg=asr_cfg,
        runtime_csv=out_root / "runtime.csv",
    )
    return 0


def cmd_vote(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    from wscaudit.merge.vote import DEFAULT_HYPS, run_vote

    out_root = Path(cfg["out_root"])
    names = tuple(n.strip() for n in str(args.hyps).split(",") if n.strip()) or DEFAULT_HYPS
    hyp_paths = {name: out_root / "hyps" / f"{name}.jsonl" for name in names}
    missing = [p for p in hyp_paths.values() if not p.is_file()]
    if missing:
        print("missing hyp files:", *missing, sep="\n  ", file=sys.stderr)
        return 2
    stem = f"vote_{args.pipeline}"
    run_vote(
        out_root / "manifest.jsonl",
        hyp_paths,
        out_root / "merged" / f"{stem}.jsonl",
        out_root / "merged" / f"{stem}.summary.json",
        split=args.split,
        limit=args.limit,
        hyp_order=names,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_paths(args.config)
    if args.command == "prepare":
        return cmd_prepare(args, cfg)
    if args.command == "asr":
        return cmd_asr(args, cfg)
    if args.command == "vote":
        return cmd_vote(args, cfg)
    return _stub(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
