import numpy as np
import json 
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor



from optimize import datasets, train


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

if __name__ == "__main__":
    
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

    tasks = []

    for filename in os.listdir("smac3_results"):
        if filename.endswith(".json"):
            with open(os.path.join("smac3_results", filename), "r") as f:
                configs = json.load(f)
                
            #Extract info best_config_loo_{dataset}_seed_{seed}.json
            loo_dataset = filename.split("_")[3]
            seed = filename.split("_")[5].split(".")[0]
            for d in datasets.values():
                for i, config in enumerate(configs):
                    tasks.append((config, d, filename, i, loo_dataset, seed))

    for d in datasets.values():
        tasks.append((default_config, d, "default", "none", "none", "none"))

    with ProcessPoolExecutor(max_workers=1 or 1) as executor:
        results = list(executor.map(_evaluate_task, tasks))

    df = pd.DataFrame(results)
    df.to_csv("final_loo_performances.csv", index=False)
