#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J finetune
#SBATCH -o finetune.%J.out
#SBATCH -e finetune.%J.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --constraint=a100

#run the application:
module purge
conda activate /ibex/user/songt/conda_envs/ontoalign
python -u finetune.py

