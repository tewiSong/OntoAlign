#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J pretrain
#SBATCH -o pretrain.%J.out
#SBATCH -e pretrain.%J.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --constraint=a100|v100

#run the application:
module purge
conda activate /ibex/user/songt/conda_envs/ontoalign
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
python pretrain.py

