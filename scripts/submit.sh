#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J dataset
#SBATCH -o deepgo_embed_dim_256.%J.out
#SBATCH -e deepgo_embed_dim_256.%J.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --constraint=v100|a100


conda activate /ibex/user/songt/conda_envs/ontoalign
python build_graphs.py
