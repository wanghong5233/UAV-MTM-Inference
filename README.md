# UAV-MTM-Inference

**Share-Aware Multi-Task DNN Partitioning and Offloading in UAV Swarms**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Work in progress.** This repository accompanies an ongoing research project. The manuscript is still being written and has **not** been submitted for peer review. APIs, configs, and scripts may change without notice.

---

## Overview

Simulation code for **share-aware multi-task DNN partitioning and offloading in UAV swarms**. The framework includes:

- **Three MTL backbones**: Split Network, MTAN, Cross-Stitch
- **Proposed method**: GNN-PPO with topology-aware partition routing
- **Baselines / ablations**: local-only execution, MLP-PPO, single-split, partition-routing without compression
- **Accuracy modeling**: quantization distortion → task accuracy degradation → surrogate fitting
- **Config-driven workflow**: train, evaluate, and plot via YAML configs and shell scripts

Large artifacts (NYUv2 data, checkpoints, logs, and experiment outputs) are **not** included in this repository; see [Setup](#setup) below.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/wanghong5233/UAV-MTM-Inference.git
cd UAV-MTM-Inference

# Linux / WSL
bash setup.sh

# Windows
powershell -ExecutionPolicy Bypass -File setup.ps1

# Or with Conda
conda env create -f environment.yml
conda activate uav-mtl
```

### 2. Prepare data

```bash
python scripts/data/0_download_nyuv2.py
python scripts/data/1_preprocess.py
python scripts/data/2_verify_data.py
```

Processed data is written under `data/` (gitignored).

### 3. Accuracy modeling (optional, before RL training)

```bash
python scripts/1_fit_accuracy.py --model split_network
python scripts/1_fit_accuracy.py --model mtan
python scripts/1_fit_accuracy.py --model crossstitch
```

### 4. Train and evaluate

```bash
# Example: proposed GNN-PPO (edit config path as needed)
python scripts/train.py --config configs/experiments/main_gnn_ppo_tmc_stable.yaml

# Batch training helpers
bash scripts/train_all.sh
bash scripts/train_multi_seed.sh

# Evaluation
python scripts/evaluate.py
```

Experiment configs live under `configs/experiments/`; algorithm and model hyperparameters are under `configs/algorithms/` and `configs/models/`.

### 5. Plotting

```bash
python scripts/plot_convergence.py
python scripts/plot_surrogate_validation.py
python scripts/plot_cross_model_e2.py
python scripts/plot_robustness_e5.py
bash scripts/run_sweep.sh          # parameter sweeps
bash scripts/run_main_table_e3.sh  # main-table experiments
```

Figures and raw metrics are saved locally under `results/` and `experiments/` (both gitignored).

---

## Project structure

```
UAV-MTM-Inference/
├── configs/
│   ├── default.yaml           # global simulation / training defaults
│   ├── algorithms/            # per-algorithm hyperparameters
│   ├── models/                # MTL backbone configs
│   └── experiments/           # combined experiment configs
├── scripts/
│   ├── train.py               # unified training entry
│   ├── evaluate.py            # unified evaluation entry
│   ├── data/                  # NYUv2 download & preprocessing
│   ├── debug/                 # smoke tests and diagnostics
│   └── plot_*.py, run_*.sh    # experiment plotting and batch runs
├── src/
│   ├── core/                  # trainer, evaluator, agent factory
│   ├── agents/                # GNN-PPO and baselines
│   ├── models/                # MTL model definitions
│   ├── accuracy_modeling/     # distortion & accuracy fitting
│   ├── data/                  # dataset loaders
│   └── utils/                 # logging, metrics, visualization
├── data/                      # local dataset (gitignored)
├── checkpoints/               # saved weights (gitignored)
├── experiments/               # training logs & metrics (gitignored)
└── results/                   # figures & eval JSON (gitignored)
```

---

## Algorithms

| Config name | Role |
|---|---|
| `gnn_ppo` | Proposed: GNN encoder + PPO for joint partition & offloading |
| `local_only` | All inference executed locally on each UAV |
| `mlp_ppo` | MLP policy + PPO (no graph structure) |
| `single_split` | Fixed single split point per task |
| `pr_no_compression_ppo` | Partition routing enabled, compression disabled |

---

## Reproducibility notes

- Python dependencies are pinned in `requirements.txt` / `environment.yml`.
- Random seeds and simulation parameters are set in `configs/default.yaml`.
- Model weights (`.pth`, `.pkl`) are tracked via Git LFS when committed locally; they are excluded from this public release.
- Re-run the scripts above after placing data and (optionally) checkpoints in the expected local paths.

---

## Citation

The associated paper is **not yet published**. A BibTeX entry will be added here after submission. If you use this code before then, please contact the authors (see below).

---

## Contact

- **GitHub**: [wanghong5233](https://github.com/wanghong5233)

Questions and bug reports are welcome via [GitHub Issues](https://github.com/wanghong5233/UAV-MTM-Inference/issues).

---

## License

Released under the MIT License.

## Acknowledgments

We thank the creators of the [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset for making their data publicly available.
