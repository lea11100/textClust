import json 
import os
import pandas as pd
from pathlib import Path
import argparse



from optimize import datasets, train


default_config = {
    "clean_interval": 100,
    "cluster_interval": 1000,
    "decay_rate": 0.005,
    "min_num_mc": 5,
    "adaptive_threshold": 1,
    "sigma": 1.0,
    "threshold_momentum": 0.0,
    "use_heuristic": 1,
    "time_gap": 1000,
    "model": "sentenceTransformers",
}


def _evaluate_task(task):
    config, dataset, config_name, config_id, loo_dataset, seed = task
    print("EVALUATING:", dataset, config_name, config_id)
    
    performance = train(config, instance=dataset)
    print("FINISHED EVALUATION:", dataset, config_name, config_id, performance)
    return {
        "dataset": dataset,
        "neg_nmi": performance["-nmi"],
        "neg_throughput": performance["-throughput"],
        "config": config_name,
        "config_id": config_id,
        "loo_dataset": loo_dataset,
        "seed": seed,
    }


def load_config(config_path):
    if config_path == "default":
        return default_config

    with open(config_path, "r") as f:
        configs = json.load(f)

    if not configs:
        raise ValueError(f"No configurations found in {config_path}")

    return configs


def evaluate_config(config, config_name="loaded_config", output_csv="final_loo_performances.csv"):
    tasks = []
    
    if config_name == "default":
        loo_dataset = "none"
        seed = "none"
    else:
        loo_dataset = config_name.split("_")[3]
        seed = config_name.split("_")[5].split(".")[0]

    for dataset_name in datasets.values():
        for i,c in enumerate(config if isinstance(config, list) else [config]):
            tasks.append((c, dataset_name, config_name, i, loo_dataset, seed))

    results = [_evaluate_task(task) for task in tasks]

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


def main(config_path=None, output_csv="final_loo_performances.csv", use_default_config=False):
    #Get folder for output CSV 
    output_folder = Path(output_csv).parent
    os.makedirs(output_folder, exist_ok=True)

    if use_default_config:
        config = default_config
        config_name = "default"
    else:
        if config_path is None:
            raise ValueError("Provide a config_path or pass --use-default-config to evaluate the built-in configuration.")
        config = load_config(f"SMAC3_results/{config_path}")
        config_name = Path(config_path).name

    return evaluate_config(config, config_name=config_name, output_csv=output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default=None, help="Path to a JSON file containing one or more configurations.")
    parser.add_argument("--use-default-config", action="store_true", help="Evaluate the built-in default configuration instead of loading a JSON file.")
    parser.add_argument("--output-csv", default="final_loo_performances.csv", help="Where to write the results CSV.")
    args = parser.parse_args()

    main(
        args.config_path,
        output_csv=args.output_csv,
        use_default_config=args.use_default_config,
    )
