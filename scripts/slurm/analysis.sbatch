#!/bin/bash
#SBATCH -A <PROJECT>
#SBATCH -t 02:00:00
#SBATCH -J wsc-analysis
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
# CPU job: no -p gpu. check `module spider python`
module load python/3.11.8
source /proj/<PROJECT>/venvs/${ENV_NAME:-core}/bin/activate
export HF_HOME=/proj/<PROJECT>/hf_cache
export WSC_DATA=/proj/<PROJECT>/data/WSC-Eval
export WSC_OUT=/proj/<PROJECT>/wsc-audit/outputs
mkdir -p logs
cd /proj/<PROJECT>/wsc-audit
python -m wscaudit "${STEP:-report}"
