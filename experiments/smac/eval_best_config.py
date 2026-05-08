import numpy as np
import json 
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor



from optimize import datasets, train


def _evaluate_task(task):
    config, dataset, config_name, loo_dataset, seed = task
    performance = train(config, instance=dataset)
    return {
        "dataset": dataset,
        "performance": performance,
        "config": config_name,
        "loo_dataset": loo_dataset,
        "seed": seed,
    }

if __name__ == "__main__":
    
    default_config = {
        "clean_interval": 100,
        "cluster_interval": 1000,
        "log_interval": 1000,
        "decay_rate": 0.005,
        "sigma": 1.0
    }

    tasks = []

    for filename in os.listdir("smac3_results"):
        if filename.endswith(".json"):
            with open(os.path.join("smac3_results", filename), "r") as f:
                config = json.load(f)
                
            #Extract info best_config_loo_{dataset}_seed_{seed}.json
            loo_dataset = filename.split("_")[3]
            seed = filename.split("_")[5].split(".")[0]
            for d in datasets.values():
                tasks.append((config, d, filename, loo_dataset, seed))

    for d in datasets.values():
        tasks.append((default_config, d, "default", "none", "none"))

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        results = list(executor.map(_evaluate_task, tasks))

    df = pd.DataFrame(results)
    df.to_csv("final_loo_performances.csv", index=False)
