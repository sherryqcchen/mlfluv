#!/usr/bin/env bash
#SBATCH --job-name=mlfluv-dlv3p-s8-lr1e4-pmiou-h100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd /users/qchen7/mlfluv/script/mlfluv
export MPLCONFIGDIR=/tmp/matplotlib-${SLURM_JOB_ID}

/users/qchen7/.micromamba/envs/sentry/bin/python MODEL_LAYER/train.py --config_path config_h100_deeplabv3plus_stride8_lr1e4_plateau_miou.yml
