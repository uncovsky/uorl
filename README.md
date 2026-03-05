# Uncertainty-based Offline Reinforcement Learning

This repository contains the code accompanying our RLC paper evaluating ensemble-based uncertainty quantification methods for offline RL, and proposing U-AWAC and U-BRAC as extensions of policy-constrained baselines with independently bootstrapped ensembles.

The unified framework is built on top of the [Unifloral](https://github.com/EmptyJackson/unifloral) library and covers the following algorithms:

| Algorithm | Reference |
|-----------|-----------|
| **SAC-N / EDAC** | [Uncertainty-Based Offline RL with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) |
| **MSG** | [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703) |
| **PBRL** | [Pessimistic Bootstrapping for Uncertainty-Driven Offline RL](https://arxiv.org/abs/2202.11566) |
| **AWAC**, **CQL** | Non-ensemble baselines |
| **U-AWAC** | Our proposed method |

The following baselines are implemented independently outside the unified framework:

| Algorithm | Reference |
|-----------|-----------|
| **ReBRAC, IQL, BC**| baselines from [Unifloral](https://github.com/EmptyJackson/unifloral) |
| **U-ReBRAC** | Our proposed extension of ReBRAC with independent bootstrap targets |


---

## Repository Structure

```
algorithms/          # Unified algorithmic framework and baselines
configs/             # Algorithm and experiment configurations
evaluation_scripts/  # Scripts to evaluate trained policies and collect rollouts
infra/               # Core framework infrastructure (models, dataset class,..)
results/             # Collected rollouts, figures, and visualization scripts 
```

### `algorithms/`
Contains the unified training loop (`unified.py`) shared by all ensemble methods, plus modified BC and ReBRAC baselines adapted from Unifloral.

### `configs/`
Two types of configs:
- **Algorithm configs** (`unified_*.yaml`) — hyperparameter settings that instantiate each algorithm within the unified framework. Refer to these to understand how the original algorithm's hyperparameters map to the framework's parameters.
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
  (notably the notebook `evaluation_plots.ipynb`) 

---

## Installation


We provide a following docker setup

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
script directly. Note that this script performs uniform sampling of training hyperparameters as in our evaluation.

```bash
python evaluation_scripts/unified_sweep_env.py \
    --algorithm  unified_uawac \
    --dataset  antmaze-medium-diverse-v2 \
    --seed 0
```

The algorithm entry corresponds to the config loaded from configs/, e.g.
"unified_msg", "unified_pbrl", "urebrac", "rebrac", "iql". The methods implemented inside the framework prepend "unified", while other baselines do not.

Use `--help` for the full list of parameters:
```bash
python evaluation_scripts/unified_sweep_env.py --help
```
All algorithms share the training logic implemented in `make_train_step()` in `algorithms/unified.py`. The loop is parameterized by critic regularization and ensemble diversity terms, specified via CLI arguments and implemented in `infra/ensemble_training/`. Algorithm-specific hyperparameter mappings are documented in the corresponding `configs/unified_*.yaml` files.
