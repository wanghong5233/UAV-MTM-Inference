# UAV-MTL-Inference

**Share-Aware Multi-Task DNN Partitioning and Offloading in UAV Swarms**

> A Deep Reinforcement Learning Framework for Multi-Task Inference Optimization in UAV Swarms

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This repository provides a comprehensive simulation framework for **share-aware multi-task DNN partitioning and offloading in UAV swarms**, designed for reproducible research and TMC publication. The framework features:

- **3 MTL Architectures**: Split Network (hard parameter sharing), MTAN (attention-based soft sharing), Cross-Stitch Networks
- **Multi-Algorithm Comparison**: GNN-PPO (proposed), Random, Greedy, and ablation variants
- **Accuracy Modeling Pipeline**: Quantization distortion → Accuracy degradation fitting → Surrogate model training
- **Fully Decoupled Design**: Algorithm-agnostic environment, plug-and-play agents via factory pattern
- **Publication-Ready**: One-click generation of IEEE-style figures and tables

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo_url>
cd UAV-MTL-Inference

# Automated setup (recommended)
bash setup.sh

# Or manually with Conda
conda env create -f environment.yml
conda activate uav-mtl

# Or with pip
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download NYUv2 dataset (semantic segmentation, depth, surface normal)
python scripts/data/0_download_nyuv2.py

# Preprocess: resize, normalize, generate splits
python scripts/data/1_preprocess.py

# Verify data integrity
python scripts/data/2_verify_data.py
```

### 3. Accuracy Modeling (Pre-Training Phase)

```bash
# Fit accuracy degradation functions A_k(D) for each MTL model
python scripts/1_fit_accuracy.py --model split_network
python scripts/1_fit_accuracy.py --model mtan
python scripts/1_fit_accuracy.py --model crossstitch

# Results saved to: experiments/accuracy_fitting/{model}/fitted_params.json
```

### 4. Train DRL Agents

```bash
# Train proposed GNN-PPO agent
python scripts/train.py --config configs/experiments/main_gnn_ppo.yaml

# Train baselines
python scripts/train.py --config configs/experiments/baseline_random.yaml
python scripts/train.py --config configs/experiments/baseline_greedy.yaml

# Run all experiments in batch
bash scripts/run_all_experiments.sh
```

### 5. Evaluation and Visualization

```bash
# Evaluate all trained agents
python scripts/evaluate.py

# Generate comparison plots and tables for paper
python scripts/compare_algorithms.py
python scripts/5_plot_results.py

# Results: results/figures/*.pdf, results/tables/*.csv
```

---

## 📐 Design Principles

1. **Decoupling**: Algorithm-environment-training separation for independent development and testing
2. **Extensibility**: Add new algorithms/models by inheriting base classes, zero core code modification
3. **Reproducibility**: Configuration-driven experiments + version control ensure full reproducibility
4. **Practicality**: Covers debugging, comparison, and visualization end-to-end
5. **Publication-Friendly**: One-click generation of IEEE TMC-ready figures and tables

---

## 📁 Project Structure

```
UAV-MTL-Inference/
│
├── .gitignore                    # Version control policy
├── .gitattributes                # Git LFS configuration (for large files)
├── README.md                     # This file
├── requirements.txt              # Core Python dependencies
├── requirements-dev.txt          # Development dependencies
├── environment.yml               # Conda environment definition
├── setup.sh                      # One-click setup script
├── Dockerfile                    # Docker containerization (optional)
│
├── configs/                      # Configuration-driven experiments
│   ├── default.yaml              # Default parameters (UAV, network, environment)
│   │
│   ├── models/                   # MTL model configurations
│   │   ├── split_network.yaml    # Split Network parameters
│   │   ├── mtan.yaml             # MTAN parameters
│   │   └── crossstitch.yaml      # Cross-Stitch parameters
│   │
│   ├── algorithms/               # Algorithm-specific hyperparameters
│   │   ├── gnn_ppo.yaml          # Proposed method: GNN+PPO
│   │   ├── random.yaml           # Random baseline
│   │   ├── greedy.yaml           # Greedy baseline
│   │   └── no_topo.yaml          # Ablation: no topology control
│   │
│   └── experiments/              # Combined experiment configs (algorithm × model)
│       ├── main_gnn_ppo.yaml     # Main experiments: GNN-PPO on 3 models
│       ├── baseline_random.yaml  # Random baseline on 3 models
│       ├── baseline_greedy.yaml  # Greedy baseline on 3 models
│       └── ablation_no_topo.yaml # Ablation study
│
├── data/                         # Datasets + Pretrained weights
│   ├── raw/                      # Raw data (after download)
│   │   └── nyuv2_raw/
│   │
│   ├── processed/                # Preprocessed data (resized, normalized)
│   │   └── nyuv2/
│   │       ├── images/
│   │       ├── labels/
│   │       └── splits.json       # Train/val/test split
│   │
│   └── pretrained/               # Pretrained MTL model weights
│       ├── split_network.pth
│       ├── mtan.pth
│       └── crossstitch.pth
│
├── src/                          # Core source code
│   │
│   ├── core/                     # Core framework (decoupling infrastructure)
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Abstract base class for algorithms
│   │   ├── agent_factory.py      # Algorithm registry (factory pattern)
│   │   ├── trainer.py            # Unified training loop
│   │   └── evaluator.py          # Unified evaluation logic
│   │
│   ├── models/                   # MTL model definitions (decoupled from algorithms)
│   │   ├── __init__.py
│   │   ├── base_mtl.py           # MTL abstract base class
│   │   ├── split_network.py      # Hard parameter sharing
│   │   ├── mtan.py               # MTAN (attention-based)
│   │   └── crossstitch.py        # Cross-Stitch Networks
│   │
│   ├── accuracy_modeling/        # Accuracy modeling (core contribution)
│   │   ├── __init__.py
│   │   ├── distortion.py         # Compute distortion D (Eq. 3.21)
│   │   ├── quantization.py       # Quantization/dequantization logic
│   │   ├── fit_accuracy.py       # Fit A_k(D) functions
│   │   ├── fit_surrogate.py      # Fit D(t) surrogate model
│   │   └── hooks.py              # PyTorch hooks for intermediate features
│   │
│   ├── env/                      # DRL environment (fully decoupled from algorithms)
│   │   ├── __init__.py
│   │   ├── uav_env.py            # Main environment (gym.Env interface)
│   │   ├── state.py              # State representation
│   │   ├── action.py             # Action space definition
│   │   ├── reward.py             # Reward calculation
│   │   └── simulator.py          # Delay/energy/channel simulation
│   │
│   ├── agents/                   # Algorithm implementations (inherit BaseAgent)
│   │   ├── __init__.py           # Auto-register all agents
│   │   ├── gnn_ppo.py            # Proposed method
│   │   ├── random_agent.py       # Random baseline
│   │   ├── greedy_agent.py       # Greedy baseline
│   │   └── no_topo_agent.py      # Ablation: no topology control
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logger.py             # Tensorboard/WandB logging
│       ├── experiment_tracker.py # Experiment tracking
│       ├── metrics.py            # Compute delay/energy/accuracy
│       ├── visualization.py      # Plotting (scatter, curves)
│       ├── checkpoint.py         # Save/load models
│       └── config_loader.py      # Configuration loading/merging
│
├── scripts/                      # Experiment scripts (unified entry points)
│   ├── train.py                  # Unified training entry (config-driven)
│   ├── evaluate.py               # Unified evaluation entry
│   ├── compare_algorithms.py     # Batch compare multiple algorithms
│   ├── 1_fit_accuracy.py         # Preprocessing: accuracy modeling fitting
│   ├── 5_plot_results.py         # Postprocessing: generate paper figures
│   ├── run_all_experiments.sh    # One-click run all experiments
│   │
│   ├── data/                     # Data processing
│   │   ├── 0_download_nyuv2.py
│   │   ├── 1_preprocess.py
│   │   └── 2_verify_data.py
│   │
│   ├── debug/                    # Quick debugging
│   │   ├── test_env.py
│   │   ├── test_agent.py
│   │   ├── test_accuracy_fit.py
│   │   └── profile_model.py
│   │
│   └── tools/                    # Auxiliary tools
│       ├── resume_training.py
│       └── clean_experiments.py
│
├── experiments/                  # Experiment results (gitignored large files)
│   │
│   ├── accuracy_fitting/         # Accuracy fitting data
│   │   ├── split_network/
│   │   │   ├── sampling_points.csv
│   │   │   ├── fitted_params.json
│   │   │   └── plots/
│   │   ├── mtan/
│   │   │   ├── sampling_points.csv
│   │   │   ├── fitted_params.json
│   │   │   └── plots/
│   │   └── crossstitch/
│   │       ├── sampling_points.csv
│   │       ├── fitted_params.json
│   │       └── plots/
│   │
│   └── training/                 # DRL training logs (organized by algorithm)
│       ├── gnn_ppo/
│       │   ├── split_network_run1/
│       │   │   ├── config.yaml   # Full config snapshot for this run
│       │   │   ├── checkpoints/
│       │   │   ├── logs/
│       │   │   └── metrics.csv
│       │   ├── mtan_run1/
│       │   └── crossstitch_run1/
│       │
│       ├── random/
│       │   ├── split_network/
│       │   ├── mtan/
│       │   └── crossstitch/
│       │
│       ├── greedy/
│       │   ├── split_network/
│       │   ├── mtan/
│       │   └── crossstitch/
│       │
│       └── no_topo/              # Ablation study
│           ├── split_network/
│           ├── mtan/
│           └── crossstitch/
│
├── checkpoints/                  # Best model weights (organized by algorithm)
│   ├── accuracy_functions/       # Fitted accuracy parameters (independent)
│   │   ├── split_network.pkl
│   │   ├── mtan.pkl
│   │   └── crossstitch.pkl
│   │
│   ├── gnn_ppo/                  # Organized by algorithm
│   │   ├── split_network_best.pth
│   │   ├── mtan_best.pth
│   │   └── crossstitch_best.pth
│   │
│   ├── random/
│   ├── greedy/
│   └── no_topo/
│
├── logs/                         # Unified runtime logs (gitignored)
│   ├── training/
│   │   ├── gnn_ppo_20250122_143022.log
│   │   ├── random_20250122_150033.log
│   │   └── greedy_20250122_152044.log
│   │
│   └── evaluation/
│       └── compare_all_20250122_160055.log
│
└── results/                      # Final figures and tables for paper (git-managed)
    ├── figures/
    │   ├── fig_accuracy_fitting.pdf
    │   ├── fig_delay_comparison.pdf     # Compare all algorithms
    │   ├── fig_energy_comparison.pdf
    │   ├── fig_pareto_frontier.pdf
    │   └── fig_ablation_study.pdf
    │
    └── tables/
        ├── tab_main_results.csv         # Main results for all algorithms
        ├── tab_ablation.csv
        └── tab_statistical_test.csv     # T-test/Wilcoxon test
```

---

## 🔧 Core Modules

### Accuracy Modeling (`src/accuracy_modeling/`)
Models the relationship between quantization distortion and task accuracy degradation. Supports arbitrary MTL topologies.

### DRL Environment (`src/env/`)
Gym-compatible environment simulating UAV swarm inference with realistic delay, energy, and channel models.

### Agent Implementations (`src/agents/`)
Plug-and-play DRL agents via factory pattern. Add new algorithms by inheriting `BaseAgent`.

### MTL Models (`src/models/`)
Three representative DAG-structured MTL architectures for scene understanding tasks (semantic segmentation, depth estimation, surface normal prediction).

---

## 🔬 Reproducibility

All experiments are **configuration-driven** and fully reproducible:

1. **Environment**: `environment.yml` locks all package versions
2. **Random Seeds**: Fixed in `configs/default.yaml`
3. **Git LFS**: Tracks large files (`.pth`, `.pkl`)
4. **Config Snapshots**: Each run saves its full config to `experiments/training/{algo}/{model}_run{id}/config.yaml`

To reproduce results:
```bash
bash scripts/run_all_experiments.sh
```

---

## 📊 Experimental Results

After running experiments, find publication-ready outputs in:

- **Figures**: `results/figures/*.pdf` (IEEE double-column format)
- **Tables**: `results/tables/*.csv` (convertible to LaTeX via pandas)
- **Raw Metrics**: `experiments/training/{algo}/{model}_run{id}/metrics.csv`

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your2025uav,
  title={Share-Aware Multi-Task DNN Partitioning and Offloading in UAV Swarms},
  author={Your Name and Advisor Name},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  note={Under Review}
}
```

---

## 🤝 Contributing

We welcome contributions! To add a new algorithm:

1. Inherit from `BaseAgent` in `src/core/base_agent.py`
2. Implement required methods: `select_action()`, `update()`, etc.
3. Register in `src/agents/__init__.py`
4. Add config in `configs/algorithms/{your_algo}.yaml`

See `src/agents/gnn_ppo.py` for a complete example.

---

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@university.edu]
- **Institution**: [Your University]
- **Lab**: [Lab Name]

For questions or collaboration inquiries, please open an issue or email directly.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work is supported by [Funding Agency]. We thank the authors of [NYUv2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) for making their data publicly available.

