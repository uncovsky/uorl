# Uncertainty-based Offline Reinforcement Learning

This repository contains the code accompanying our RLC paper evaluating ensemble-based uncertainty quantification methods for offline RL, and proposing U-AWAC and U-BRAC as extensions of policy-constrained baselines 7
with independently bootstrapped ensembles.

The framework unifies the following algorithms under a common implementation:

| Algorithm | Reference |
|-----------|-----------|
| **SAC-N / EDAC** | [Uncertainty-Based Offline RL with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) |
| **MSG** | [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703) |
| **PBRL** | [Pessimistic Bootstrapping for Uncertainty-Driven Offline RL](https://arxiv.org/abs/2202.11566) |
| **AWAC**, **CQL** | Non-ensemble baselines |
| **U-AWAC**, **U-BRAC** | Our proposed methods |

Built on top of the [Unifloral](https://github.com/EmptyJackson/unifloral) library.

---

## Repository Structure

```
algorithms/          # Unified algorithmic framework and baselines
configs/             # Algorithm and experiment configurations
data/                # Dataset utilities (D4RL and Minari)
evaluation_scripts/  # Scripts to evaluate trained policies and collect rollouts
infra/               # Core framework infrastructure
results/             # Collected rollouts, figures, and visualization scripts
setup.py             # Package installation
```

### `algorithms/`
Contains the unified training loop (`unified.py`) shared by all ensemble methods, plus modified BC and ReBRAC baselines adapted from Unifloral.

### `configs/`
Two types of configs:
- **Algorithm configs** (`unified_*.yaml`) — hyperparameter settings that instantiate each algorithm within the unified framework. Refer to these to understand how original algorithm hyperparameters map to framework parameters.
- **Experiment configs** (`eval_*.yaml`) — wandb sweep configs defining the hyperparameter spaces used to reproduce experiments from the paper.

### `infra/`
Core infrastructure shared across algorithms:
- `models/` — actor and critic networks (from Unifloral, extended with randomized prior network support)
- `ensemble_training/` — critic regularization losses (MSG, EDAC, CQL, etc.)
- `dataset/` — dataset wrapper supporting both D4RL and Minari environments
- `checkpoints/` — model checkpointing utilities
- `utils/` — logging, checkpointing

### `results/`
Contains all experimental data and figures from the paper:
- `rollouts/` — evaluation rollouts organized by algorithm and dataset
- Visualization scripts to reproduce all figures from the paper
  (notably evaluation_plots.ipynb) 

---

## Installation


Or use the provided Docker setup (recommended for full reproducibility):

```bash
make build       # Build the Docker image
make up          # Launch with GPU (requires nvidia-container-toolkit)
make up-cpu      # Launch without GPU
make down        # Stop and remove the container
```
---

## Running Experiments

Reproducing the paper experiments can be done by running the wandb sweeps in `configs/`
To launch a sweep:

```bash
wandb sweep configs/eval_full_evaluation.yaml
wandb agent <SWEEP_ID>  # Run the agent with the sweep ID printed by the previous command
```
Note that this requires a wandb account and API key. 


Alternatively, a single run can be launched without wandb using the evaluation
script directly:

```bash
python evaluation_scripts/unified_sweep_env.py \
    --algorithm  unified_uawac \
    --dataset  antmaze-medium-diverse-v2 \
    --seed 0
```

The algorithm entry corresponds to the config loaded from configs/, e.g.
"unified_msg", "unified_pbrl", "urebrac", "rebrac", "iql".

Use `--help` for the full list of parameters:
```bash
python evaluation_scripts/unified_sweep_env.py --help
```
All algorithms share the training logic implemented in `make_train_step()` in `algorithms/unified.py`. The loop is parameterized by critic regularization and ensemble diversity terms, specified via CLI arguments and implemented in `infra/ensemble_training/`. Algorithm-specific hyperparameter mappings are documented in the corresponding `configs/unified_*.yaml` files.
