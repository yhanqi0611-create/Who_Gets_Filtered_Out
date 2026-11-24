#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper"
if ! command -v latexmk >/dev/null 2>&1 && ! command -v pdflatex >/dev/null 2>&1; then
  echo "TeX not installed; skipping build"
  exit 0
fi
if [[ ! -f tacl2021v1.sty ]]; then
  echo "Run scripts/fetch_tacl_style.sh"
  exit 1
fi
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode main.tex
else
  pdflatex -interaction=nonstopmode main.tex
  bibtex main || true
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
fi
if command -v pdfinfo >/dev/null 2>&1 && [[ -f main.pdf ]]; then
  pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
  echo "page count: $pages"
  if [[ "$pages" -gt 8 ]]; then
    echo "warning: page count $pages exceeds 8 before checking references"
  fi
fi
