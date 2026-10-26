#!/bin/bash
#SBATCH -A <PROJECT>
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 04:00:00
#SBATCH -J wsc-merge
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
# check `module spider python`
module load python/3.11.8
source /proj/<PROJECT>/venvs/${ENV_NAME:-llm}/bin/activate
export HF_HOME=/proj/<PROJECT>/hf_cache
export WSC_DATA=/proj/<PROJECT>/data/WSC-Eval
export WSC_OUT=/proj/<PROJECT>/wsc-audit/outputs
mkdir -p logs
cd /proj/<PROJECT>/wsc-audit
python -m wscaudit merge --pipeline "${PIPELINE:-P0}" --device cuda:0
