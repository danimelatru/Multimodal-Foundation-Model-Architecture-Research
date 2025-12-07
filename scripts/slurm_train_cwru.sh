#!/bin/bash
#SBATCH --job-name=train_cwru
#SBATCH --output=train_cwru_%j.out
#SBATCH --error=train_cwru_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1 

module load python/3.9.10/gcc-11.2.0
source ~/moment_env_py39/bin/activate
cd ~/projects/moment

python train_cwru_moment.py

