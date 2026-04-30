import argparse
import sys
import os
import json
import shutil

from ConfigSpace import Configuration, ConfigurationSpace
from sentence_transformers import SentenceTransformer
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.read_and_write import pcs_new as pcs

from textClust.algorithms.presets import get_embClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.utils.eval import eval_against_truth

datasets = {
    "Semantic10-T": "examples/data/Semantic10-T.json",
    "Semantic40-T": "examples/data/Semantic40-T.json",
    "Semantic5-T": "examples/data/Semantic5-T.json",
    "NT": "examples/data/NT.json",
    "NTS": "examples/data/NTS.json",
    "NTSO": "examples/data/NTSO.json",
    "SO-T": "examples/data/SO-T.json",
    "News-T": "examples/data/News-T.json",
    "Trends-T": "examples/data/Trends-T.json",
    "Tweets-T": "examples/data/Tweets-T.json",
}

embedding_model_name = 'ibm-granite/granite-embedding-107m-multilingual'
embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

def train(config: Configuration, seed: int = 1, instance:str = "") -> float:
    algorithm = get_embClust(data_loader=StaticJsonWrapper(instance),
                             embedding_model=embedding_model,
                             log_interval=1000,
                             **config)
    algorithm.verbose = False
    
    algorithm.start()
    result = eval_against_truth(
        initial_text_assignments=algorithm.history_manager.initial_text_assignments,
        final_text_assignments=algorithm.history_manager.get_final_assignments(),
        true_assignments=algorithm.history_manager.true_assignments
    )
    return -1 * result.get("v_measure_final", 0.0)
    

# Add args for seed and instance selection if needed
args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=1, help="Random seed for SMAC (default: 1).")
args.add_argument("--loo_dataset", type=str, default="none", help="Dataset to leave out for evaluation (default: 'none').")
    
if __name__ == "__main__":
    parsed_args = args.parse_args()

    if parsed_args.loo_dataset == "none":
        selected_datasets = list(datasets.values())
    else:
        selected_datasets = [path for name, path in datasets.items() if name != parsed_args.loo_dataset]

    with open(f'experiments/smac/params.pcs', 'r') as f:
        configuration = pcs.read(f)

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configuration, 
        instances=selected_datasets,
        instance_features={name: [i] for i, name in enumerate(selected_datasets)},  # Map instance names to indices
        deterministic=True,
        walltime_limit=60 * 60 * 24 * 7 ,
        n_trials=250,
        seed=parsed_args.seed,
        output_directory=os.path.join("smac3_output", f"loo_{parsed_args.loo_dataset}_seed_{parsed_args.seed}")
    )

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()

    print("Best found configuration:", incumbent)
    
    #Store the best configuration in a JSON file for later use
    best_config_path = os.path.join("smac3_results", f"best_config_loo_{parsed_args.loo_dataset}_seed_{parsed_args.seed}.json")
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump(dict(incumbent.items()), f, indent=4)