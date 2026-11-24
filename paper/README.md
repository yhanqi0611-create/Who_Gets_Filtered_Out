# Paper skeleton (Prompt 17)

Read-only over `results/` and metrics/; every number in the paper is a generated macro.

## Local build

```bash
bash scripts/fetch_tacl_style.sh
python -m wscaudit paper-tables --snapshot
bash scripts/build_paper.sh
```

`python -m wscaudit paper-tables` regenerates `numbers.tex`, tables, figures copies, and `PAPER_PROVENANCE.txt` from frozen inputs. That should never be needed after the first snapshot; a hash mismatch fails first.

`make` in this folder runs `latexmk -pdf main.tex`. Compiler: pdfLaTeX.

## Overleaf

```bash
cd paper
zip -r ../paper_overleaf.zip . -x '_kit/*'
```

Then *New Project → Upload Project*. Set the compiler to **pdfLaTeX** and TeX Live 2023 or later. Overleaf provides `CJKutf8` and the `gbsn` font.
