"""Shared text normalisation for refs and hypotheses.

Order is a documented choice (the WSC paper does not specify it):
NFKC → SenseVoice tags → punctuation → OpenCC t2s → Latin lower-case →
whitespace → optional Arabic digits → optional spelling equivalences.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml
from opencc import OpenCC
from zhon.hanzi import punctuation as CJK_PUNCT

try:
    import cn2an
except ImportError:  # pragma: no cover
    cn2an = None  # type: ignore[assignment]

logger = logging.getLogger("wscaudit.normalize")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")
_ARABIC_DIGITS = re.compile(r"[0-9]+")

DEFAULT_NORM: dict[str, Any] = {
    "nfkc": True,
    "strip_sensevoice_tags": True,
    "strip_punctuation": True,
    "opencc_t2s": True,
    "latin_lower": True,
    "strip_whitespace": True,
    "arabic_digits_to_cn": True,
    "apply_equivalences": False,
    "equivalences_path": "resources/equivalences.tsv",
}


def load_normalization_cfg(models_yaml: Path | None = None) -> dict[str, Any]:
    path = models_yaml or (_REPO_ROOT / "configs" / "models.yaml")
    cfg = dict(DEFAULT_NORM)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded.get("normalization") or {})
    return cfg


@lru_cache(maxsize=1)
def _t2s() -> OpenCC:
    return OpenCC("t2s")


def _is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P") or ch in CJK_PUNCT


def load_equivalences(path: Path) -> list[tuple[str, str]]:
    """Load ``variant<TAB>canonical`` rows; longest variants first."""
    pairs: list[tuple[str, str]] = []
    if not path.is_file():
        return pairs
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.lower().startswith("variant\t"):
                continue
            if "\t" not in raw:
                continue
            variant, canonical = raw.split("\t", 1)
            variant, canonical = variant.strip(), canonical.strip()
            if variant:
                pairs.append((variant, canonical))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _convert_digits(text: str, digit_log: list[tuple[str, str]] | None) -> str:
    if cn2an is None:
        raise RuntimeError("cn2an is required for arabic_digits_to_cn")

    def repl(match: re.Match[str]) -> str:
        src = match.group(0)
        try:
            dst = cn2an.an2cn(src)
        except Exception as exc:  # noqa: BLE001 — log and keep the digits
            logger.warning("cn2an.an2cn(%r) failed: %s", src, exc)
            return src
        logger.info("arabic digits %s -> %s", src, dst)
        if digit_log is not None:
            digit_log.append((src, dst))
        return dst

    return _ARABIC_DIGITS.sub(repl, text)


def _apply_equivalences(text: str, pairs: list[tuple[str, str]]) -> str:
    for variant, canonical in pairs:
        if variant in text:
            text = text.replace(variant, canonical)
    return text


def light_normalize(text: str) -> str:
    """NFKC + Latin lower-case + strip whitespace. Used to classify ref diffs."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch.lower() if ch.isascii() and ch.isalpha() else ch for ch in text)
    return "".join(ch for ch in text if not ch.isspace())


def normalize(
    text: str,
    cfg: dict[str, Any] | None = None,
    *,
    digit_log: list[tuple[str, str]] | None = None,
    equivalences: list[tuple[str, str]] | None = None,
) -> str:
    """Normalise a ref or hypothesis. ``cfg`` keys match ``configs/models.yaml``."""
    cfg = {**DEFAULT_NORM, **(cfg or {})}
    out = text if text is not None else ""

    if cfg.get("nfkc", True):
        out = unicodedata.normalize("NFKC", out)
    if cfg.get("strip_sensevoice_tags", True):
        out = _SENSEVOICE_TAG.sub("", out)
    if cfg.get("strip_punctuation", True):
        out = "".join(ch for ch in out if not _is_punctuation(ch))
    if cfg.get("opencc_t2s", True):
        out = _t2s().convert(out)
    if cfg.get("latin_lower", True):
        out = "".join(
            ch.lower() if ch.isascii() and ch.isalpha() else ch for ch in out
        )
    if cfg.get("strip_whitespace", True):
        out = "".join(ch for ch in out if not ch.isspace())
    if cfg.get("arabic_digits_to_cn", True):
        out = _convert_digits(out, digit_log)
    if cfg.get("apply_equivalences", False):
        if equivalences is None:
            eq_path = Path(cfg.get("equivalences_path") or "resources/equivalences.tsv")
            if not eq_path.is_absolute():
                eq_path = _REPO_ROOT / eq_path
            equivalences = load_equivalences(eq_path)
        out = _apply_equivalences(out, equivalences)
    return out


def report_ref_diffs(
    refs: Iterable[tuple[str, str]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare full ``normalize`` vs whitespace+Latin-case-only on gold refs."""
    cfg = {**DEFAULT_NORM, **(cfg or {})}
    n = n_any = n_light_only = n_other = 0
    other_examples: list[dict[str, str]] = []
    for utt_id, ref in refs:
        n += 1
        full = normalize(ref, cfg)
        light = light_normalize(ref)
        if full == ref:
            continue
        n_any += 1
        if full == light:
            n_light_only += 1
        else:
            n_other += 1
            if len(other_examples) < 20:
                other_examples.append(
                    {"utt_id": utt_id, "ref": ref, "light": light, "norm": full}
                )
    summary = {
        "n": n,
        "changed": n_any,
        "whitespace_or_latin_case_only": n_light_only,
        "other": n_other,
        "other_examples": other_examples,
    }
    print("=== ref normalisation check ===")
    print(f"n={n}  changed={n_any}  whitespace/latin-case only={n_light_only}  other={n_other}")
    if n_other:
        print("other changes (not only whitespace / Latin case); listing examples, not editing refs:")
        for ex in other_examples:
            print(f"  {ex['utt_id']}: {ex['ref']!r} -> {ex['norm']!r}")
    else:
        print("refs change only by whitespace and/or Latin case (plus NFKC).")
    return summary


def refs_from_data_root(data_root: Path) -> list[tuple[str, str]]:
    from wscaudit.data import SPLITS, parse_text_file

    rows: list[tuple[str, str]] = []
    for split in SPLITS:
        path = Path(data_root) / "WSC-Eval-ASR" / split / "text"
        if path.is_file():
            rows.extend(parse_text_file(path))
    return rows


def refs_from_manifest(manifest: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append((rec["utt_id"], rec["ref"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    """``python -m wscaudit.normalize`` — acceptance check on gold refs."""
    import argparse
    import os

    from wscaudit.cli import load_paths

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Check normalisation on WSC-Eval refs")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = load_paths(args.config)
    cfg = load_normalization_cfg()
    if args.manifest:
        refs = refs_from_manifest(args.manifest)
    else:
        out_manifest = Path(paths["out_root"]) / "manifest.jsonl"
        data_root = Path(os.environ.get("WSC_DATA") or paths["data_root"])
        if out_manifest.is_file():
            refs = refs_from_manifest(out_manifest)
        else:
            refs = refs_from_data_root(data_root)
    if not refs:
        print("no refs found (set WSC_DATA or pass --manifest)", file=sys.stderr)
        return 2
    report_ref_diffs(refs, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
