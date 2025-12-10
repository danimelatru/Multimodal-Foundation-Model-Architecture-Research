#!/bin/bash
#SBATCH --job-name=train_cwru
#SBATCH --output=logs/train_cwru_%j.out
#SBATCH --error=logs/train_cwru_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc
conda activate moment_env_py39

PROJECT_ROOT=/gpfs/workdir/fernandeda/projects/moment
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

echo "[DEBUG SLURM] Hostname: $(hostname)"
echo "[DEBUG SLURM] Which python: $(which python)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[DEBUG SLURM] nvidia-smi:"
nvidia-smi || echo "[DEBUG SLURM] nvidia-smi failed"

python -u - << 'EOF'
import os, torch
print("[DEBUG PY] torch file:", torch.__file__)
print("[DEBUG PY] torch version:", torch.__version__)
print("[DEBUG PY] cuda available:", torch.cuda.is_available())
print("[DEBUG PY] cuda device count:", torch.cuda.device_count())
print("[DEBUG PY] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
EOF

python -u scripts/train_cwru_moment.py