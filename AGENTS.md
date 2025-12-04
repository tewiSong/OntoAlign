# Repository Guidelines

## Project Structure & Module Organization
This repository is script-driven: `build_graphs.py` ingests OWL files from `datasets/owls_align_pretrain/` and writes serialized `torch_geometric` graphs into `graphs/`; `dataset.py` and `model.py` define the dataloader plus `OntoAlignEncoder`. Training entry points live in `pretrain.py` and `finetune.py`, while lexical heuristics and evaluation helpers live in `run_baselines.py` and `alignment_utils.py`. Configuration is stored under `config/`, runtime artifacts go to `checkpoints/`, and Slurm-ready wrappers are in `scripts/`. Keep new domain-specific tooling inside `scripts/` (for jobs) or a focused module so that the top-level remains discoverable.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — provision PyTorch, rdflib, transformers, and related tooling in a clean virtual env.
- `python build_graphs.py` — encodes every OWL in `datasets/owls_align_pretrain/` according to `config/pretrain.yaml` and stores `.pt` tensors in `graphs/`.
- `python check_data.py` — sanity-checks representative graphs (keys, labels, IRIs) before training.
- `python pretrain.py` / `python finetune.py` — launch the contrastive pretraining or alignment finetuning loops using the YAML defaults; edit `config/*.yaml` before running or override via `pretrain.train("config/custom.yaml")`.
- `python run_baselines.py` — runs lexical baselines on `datasets/oaei_anatomy` and prints precision/recall via `AlignmentEvaluator`.
- `sbatch scripts/run_pretrain.sh` or `scripts/run_finetune.sh` — submit the same flows to SLURM with the shared `/ibex/user/songt/conda_envs/ontoalign` environment.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` functions, `CamelCase` classes, and descriptive module names (e.g., `alignment_utils.py`). Keep tensors explicitly typed (`torch.float32`, `torch.long`) and pass configuration via dictionaries or dataclasses rather than implicit globals. YAML keys stay lowercase with underscores, and CLI/log strings should describe the graph or dataset being processed.

## Testing Guidelines
conda activate /ibex/user/songt/conda_envs/ontoalign

srun --jobid=42830856 nvidia-smi

## Rules
- Never use fallback
- Never simplify my request
- Never use placeholder
