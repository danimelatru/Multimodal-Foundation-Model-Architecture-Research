#!/bin/bash
#SBATCH --job-name=train_cwru_scratch
#SBATCH --output=logs/train_cwru_scratch_%j.out
#SBATCH --error=logs/train_cwru_scratch_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge

source ~/.bashrc

conda activate moment_env_py39

PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

python -u scripts/train_cwru_moment_scratch.py