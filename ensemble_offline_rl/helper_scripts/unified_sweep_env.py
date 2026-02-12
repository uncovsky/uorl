import argparse
import yaml
import random
import importlib

# Maps datasets to seeds for reproducibility in hyperparam sampling
DATASET_TO_SAMPLING_SEED = {
    "hopper-medium-v2": 101,
    "halfcheetah-medium-expert-v2": 102,
    "walker2d-medium-replay-v2": 103,
    "pen-human-v1": 201,
    "pen-cloned-v1": 202,
    "pen-expert-v1": 203,
    "antmaze-large-diverse-v2": 301,
    "maze2d-large-v1": 302,
    "kitchen-mixed-v0": 401,
}

def sample_config(config_dict):
    sampled_config = {}
    data = None
    for key, value in config_dict.items():
        # preprocess
        if "values" in value:
            data = value["values"]
        elif "value" in value:
            data = value["value"]
        elif "low" in value and "high" in value:
            sampled_config[key] = random.uniform(value["low"], value["high"])
        else:
            print(f"Invalid config for key {key}: {value}")
            return None

        # Sample from values
        if isinstance(data, list):
            sampled_config[key] = random.choice(data)
        else:
            # only one value
            sampled_config[key] = data
    return sampled_config


def load_config(config_path):
    try:
        with open(f"configs/{config_path}.yaml", "r") as f:
            config = yaml.safe_load(f)

    except Exception as e:
        print(f"Error loading config file: {e}")

    # Can be wandb sweep yaml, or just plain yaml with parameters
    if "parameters" in config:
        parameters = config["parameters"]
    else:
        parameters = config
    return parameters

def cast_to_native_types(config):
    """Convert all numeric values to native Python types for JAX compatibility."""

    casted = {}
    for key, value in config.items():
        if isinstance(value, str):
            # Try to convert string to float (handles scientific notation like '1e-3')
            try:
                # Check if it looks like a number
                if 'e' in value.lower() or '.' in value or value.replace('-', '').isdigit():
                    # Try float first (handles '1e-3', '0.001', etc.)
                    casted[key] = float(value)
                else:
                    casted[key] = value
            except ValueError:
                # Not a number, keep as string
                casted[key] = value
        elif isinstance(value, (int, float)):
            # Explicitly cast to Python native types
            if isinstance(value, float):
                casted[key] = float(value)
            else:
                casted[key] = int(value)
        elif isinstance(value, bool):
            casted[key] = bool(value)
        else:
            casted[key] = value
    return casted


def load_train_fn(module_name, function_name):
    try:
        module = importlib.import_module(module_name)
        train_fn = getattr(module, function_name)
        args = getattr(module, "Args")
        return train_fn, args
    except Exception as e:
        print(f"Error loading training function: {e}")
        return None, None


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Run an experiment with the unified algorithm.")

    argument_parser.add_argument("--runs", type=int, default=20, help="The number of runs to train for")

    argument_parser.add_argument("--num_updates", type=int, default=1000000, help="The number of epochs to train for")

    argument_parser.add_argument("--num_eval_eps", type=int, default=1000, help="The number of episodes to evaluate for")

    argument_parser.add_argument("--dataset_source", type=str, default="d4rl",
                                 help="The source of the dataset.")

    argument_parser.add_argument("--dataset", type=str,
                                 default="hopper-medium-v2", 
                                 help="The name of the dataset to use.")

    argument_parser.add_argument("--algorithm", type=str,
                                 default="unified_sacn",
                                 help="Config name of the algorithm to train with.")

    # wandb entity and project
    argument_parser.add_argument("--entity", type=str, default=None, help="The wandb entity to log to.")
    argument_parser.add_argument("--project", type=str, default=None, help="The wandb project to log to.")
    args = argument_parser.parse_args()

    """
        Get algo config, sample hyperparams, run N training loops
    """
    parameters = load_config(args.algorithm)

    if args.dataset not in DATASET_TO_SAMPLING_SEED:
        raise ValueError(f"Add a seed for {args.dataset} in DATASET_TO_SAMPLING_SEED.")

    seed = DATASET_TO_SAMPLING_SEED[args.dataset]
    random.seed(seed)

    """
        dynamically load the training method
    """
    if "unified" in args.algorithm:
        train_fn, Args = load_train_fn("algorithms.unified", "train")
    else:
        train_fn, Args = load_train_fn(f"algorithms.{args.algorithm}", "train")

    print(f"Running {args.runs} runs of algorithm {args.algorithm}:")

    for seed, run in enumerate(range(1, args.runs + 1)):

        # Sample random hyperparameters
        sampled_config = sample_config(parameters)

        # Rewrite dataset settting + train seed
        sampled_config["algorithm"] = args.algorithm
        sampled_config["dataset_name"] = args.dataset
        sampled_config["seed"] = seed
        sampled_config["num_updates"] = args.num_updates
        sampled_config["eval_final_episodes"] = args.num_eval_eps

        # For unified, we need to specify the dataset source (e.g. d4rl, etc.)
        if "dataset_source" in sampled_config:
            sampled_config["dataset_source"] = args.dataset_source

        # wandb
        sampled_config["wandb_team"] = args.entity
        sampled_config["wandb_project"] = args.project
        sampled_config = cast_to_native_types(sampled_config)

        args = Args(**sampled_config)
        train_fn(args)

    print("All runs completed.")
