from algorithms.unified import train, Args
import argparse
import yaml
import random

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
            sampled_config[key] = data


    return sampled_config

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Run an experiment with the unified algorithm.")

    argument_parser.add_argument("--runs", type=int, default=20, help="The number of runs to train for")

    argument_parser.add_argument("--num_updates", type=int, default=1000000, help="The number of epochs to train for")


    argument_parser.add_argument("--sampling_seed", type=int, default=42,
                                 help="The seed used for hyperparameter sampling")

    argument_parser.add_argument("--dataset_source", type=str, default="d4rl",
                                 help="The source of the dataset.")

    argument_parser.add_argument("--dataset", type=str,
                                 default="hopper-medium-v2", 
                                 help="The name of the dataset to use.")

    argument_parser.add_argument("--algorithm", type=str,
                                 default="sac_n",
                                 help="Config name of the algorithm to train with.")

    # wandb entity and project
    argument_parser.add_argument("--entity", type=str, default=None, help="The wandb entity to log to.")
    argument_parser.add_argument("--project", type=str, default=None, help="The wandb project to log to.")
    args = argument_parser.parse_args()


    """
        Get algo config, sample hyperparams, run N training loops
    """

    with open(f"configs/{args.algorithm}.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Can be wandb sweep yaml, or just plain yaml with parameters
    if "parameters" in config:
        parameters = config["parameters"]
    else:
        parameters = config

    # sample from the parameters
    random.seed(args.sampling_seed)
    
    print(f"Running {args.runs} runs of algorithm {args.algorithm}:")

    for seed, run in enumerate(range(1, args.runs + 1)):

        # Sample random hyperparameters
        sampled_config = sample_config(parameters)

        # Rewrite dataset settting + train seed
        sampled_config["algorithm"] = args.algorithm
        sampled_config["dataset_source"] = args.dataset_source
        sampled_config["dataset_name"] = args.dataset
        sampled_config["seed"] = seed

        sampled_config["num_updates"] = args.num_updates

        # wandb
        sampled_config["wandb_team"] = args.entity
        sampled_config["wandb_project"] = args.project

        args = Args(**sampled_config)
        for key, value in sampled_config.items():
            print(f"{key}: {value}")
        train(args)

    print("All runs completed.")
