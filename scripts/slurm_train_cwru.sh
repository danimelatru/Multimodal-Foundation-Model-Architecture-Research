#!/bin/bash
#SBATCH --job-name=train_cwru
#SBATCH --output=logs/train_cwru_%j.out
#SBATCH --error=logs/train_cwru_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
source ~/.bashrc
conda activate moment_env_py39

cd /gpfs/workdir/fernandeda/projects/moment

python -u scripts/train_cwru_moment.py
