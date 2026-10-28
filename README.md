# wsc-audit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yhanqi0611-create/Who_Gets_Filtered_Out/blob/main/notebooks/colab_runner.ipynb)

Re-run the WenetSpeech-Chuan automatic labelling pipeline on the human-transcribed **WSC-Eval-ASR** set (`ASLP-lab/WSC-Eval`) and audit who gets filtered out.

GitHub: https://github.com/yhanqi0611-create/Who_Gets_Filtered_Out

Code is written to run **unchanged** on:

- UPPMAX Pelle (SLURM `sbatch`)
- Google Colab or a rented GPU (`bash scripts/run_local.sh`)

## Layout

```
configs/          paths, models, pipelines
src/wscaudit/     CLI and library (src layout)
scripts/slurm/    Pelle jobs
scripts/run_local.sh
notebooks/colab_runner.ipynb
resources/        marker list and spelling equivalences (you fill these)
```

## Paths

Defaults live in `configs/paths.yaml`. Override with environment variables:

| Variable   | Config key  | Meaning                          |
|------------|-------------|----------------------------------|
| `WSC_DATA` | `data_root` | Local WSC-Eval snapshot          |
| `WSC_OUT`  | `out_root`  | JSONL / CSV / figure outputs     |
| `HF_HOME`  | `hf_home`   | Hugging Face cache               |

Every step is `python -m wscaudit <step> ...`. Steps read/write JSONL keyed by `utt_id` and skip utterances already written (once implemented).

## Local / Colab / rented GPU

```bash
python3.11 -m venv .venv-core
source .venv-core/bin/activate
pip install -U pip
pip install -e .
# extra env for a recogniser, e.g.:
# pip install -r requirements-sensevoice.txt

export WSC_DATA="$PWD/data/WSC-Eval"
export WSC_OUT="$PWD/outputs"
bash scripts/run_local.sh prepare
```

`bash scripts/run_local.sh all` runs the full sequence (ASR models need their own venvs if dependencies conflict). Copy `outputs/` back with `rsync`; commit only the small `results/` tables, not wav or JSONL dumps.

Colab: open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) from this GitHub repo (or *File → Upload notebook* in Colab). It clones `Who_Gets_Filtered_Out`, mounts Drive for `WSC_DATA` / `WSC_OUT`, and keeps `HF_HOME` on the local disk (`/content/hf_cache`). Putting the Hugging Face cache on Drive often corrupts SentencePiece `.bpe.model` files.

## UPPMAX Pelle

Replace `<PROJECT>` in the `scripts/slurm/*.sbatch` files with the SUPR/UPPMAX project id. Put the clone and venvs on project storage, not `$HOME`.

```bash
git clone <this-repo> /proj/<PROJECT>/wsc-audit
module load python/3.11.8   # confirm with: module spider python
python -m venv /proj/<PROJECT>/venvs/core
# one venv per recogniser family if needed
export HF_HOME=/proj/<PROJECT>/hf_cache
sbatch --export=ALL,MODEL=sensevoice,ENV_NAME=asr-sensevoice scripts/slurm/asr.sbatch
sbatch --export=ALL,PIPELINE=P0,ENV_NAME=llm scripts/slurm/merge.sbatch
sbatch --export=ALL,STEP=report,ENV_NAME=core scripts/slurm/analysis.sbatch
```

Analysis is a CPU job (no `-p gpu`).

## Environments

| File                         | Use                         |
|------------------------------|-----------------------------|
| `requirements-core.txt`      | CLI, metrics, stats         |
| `requirements-sensevoice.txt`| FunASR SenseVoice-Small     |
| `requirements-qwen3asr.txt`  | Qwen3-ASR                   |
| `requirements-whisper.txt`   | Whisper-large-v3            |
| `requirements-llm.txt`       | Qwen3 merge (vLLM)          |
| `third_party/FireRedASR`     | clone; see `third_party/README.md` |

## CLI

```text
python -m wscaudit --help
python -m wscaudit {prepare,asr,merge,vote,confidence,cer,markers,calibration,stats,report}
```

`prepare` snapshots `ASLP-lab/WSC-Eval` (`WSC-Eval-ASR/**` only) into `WSC_DATA`, rebuilds wav paths (the shipped `wav.scp` is ignored), and writes `WSC_OUT/manifest.jsonl`. Re-run is safe: Hugging Face resumes. Use `--skip-download` if the files are already there.

Normalisation (NFKC, SenseVoice tags, punctuation, OpenCC t2s, Latin case, whitespace, optional digits/equivalences) lives in `wscaudit.normalize`. Check gold refs with:

```bash
python -m wscaudit.normalize
# or: python -m wscaudit.normalize --manifest "$WSC_OUT/manifest.jsonl"
```

SenseVoice:

```bash
python -m wscaudit asr --model sensevoice --split Hard --limit 20
# full Hard: drop --limit; full set: --split all
# writes $WSC_OUT/hyps/sensevoice.jsonl (idempotent) and $WSC_OUT/runtime.csv
```

FireRedASR-AED-L (separate env; do not put `HF_HOME` on Google Drive):

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git third_party/FireRedASR
pip install -r third_party/FireRedASR/requirements.txt
python -m wscaudit asr --model firered --split Hard --limit 20 --device auto
# then drop --limit; then --split Easy
# writes $WSC_OUT/hyps/firered.jsonl  (batch_size default 8 on T4)
```

Qwen3-ASR / Whisper wrappers are still stubs.

## Licence / data

WSC-Eval is Apache-2.0 on Hugging Face. We do not redistribute the audio.
