#!/bin/bash
#SBATCH --job-name=comp_dt
#SBATCH --output=logs/comp_dt_%j.out
#SBATCH --error=logs/comp_dt_%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge

source ~/.bashrc
conda activate moment_env_py39

PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

python -u scripts/compare_digital_twin.py