import argparse
import sys
import os
import json
import shutil
import sys
from pathlib import Path
import time
from typing import List
import numpy as np



from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster

ROOT = Path(__file__).resolve().parent  # directory of this file
sys.path.append(str(ROOT / "../ext_lib/mosmac"))

from ConfigSpace import Configuration, ConfigurationSpace
from sentence_transformers import SentenceTransformer
from smac.facade import MultiObjectiveFacade
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.read_and_write import pcs_new as pcs

from textClust.algorithms.presets import get_embClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.utils.eval import eval_against_truth
from textClust.offline.clusterer import SimpleMacroClusterer
from textClust.history.manager import HistoryManager
from textClust.online.clusterer import Microclusterer

datasets = {
    "Semantic10-T": "examples/data/Semantic10-T.json",
    "Semantic40-T": "examples/data/Semantic40-T.json",
    "Semantic5-T": "examples/data/Semantic5-T.json",
    "NT": "examples/data/NT.json",
    "NTS": "examples/data/NTS.json",
    "NTSO": "examples/data/NTSO-T.json",
    "SO-T": "examples/data/SO-T.json",
    "News-T": "examples/data/News-T.json",
    "Trends-T": "examples/data/Trends-T.json",
    "Tweets-T": "examples/data/Tweets-T.json",
}

# embedding_model_name = 'ibm-granite/granite-embedding-107m-multilingual'
# embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

models = {
    "sentenceTransformers": SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device="cpu"),
    "minishlab": SentenceTransformer('minishlab/potion-base-32M', device="cpu"),
    "clips": SentenceTransformer('clips/e5-base-trm-nl', device="cpu"),
    "maniac": SentenceTransformer('ManiacLabs/miniac-embed', device="cpu"),
    }

class ThroughputLimitHistoryManager(HistoryManager):
    def __init__(self, lower_bound: float, num_points: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_time = time.time()
        self.lower_bound = lower_bound
        self.num_points = num_points

    def snapshot(self, current_time: float, clusterer: Microclusterer):
        super().snapshot(current_time, clusterer)
        elapsed_time = time.time() - self.current_time
        throughput = self.num_points / elapsed_time if elapsed_time > 0 else float('inf')
        if throughput < self.lower_bound:
            raise RuntimeError(f"Throughput {throughput:.2f} is below the lower bound of {self.lower_bound}. Terminating optimization.")
        self.current_time = time.time()  # Reset timer for next snapshot

def train(config: Configuration, seed: int = 1, instance:str = "") -> dict[str, float]:
    embedding_model = models.get(config.get("model", "sentenceTransformers"), models["sentenceTransformers"])
    algorithm = get_embClust(data_loader=StaticJsonWrapper(instance),
                             embedding_model=embedding_model,
                             log_interval=1000,
                             **config)
    algorithm.verbose = False
    algorithm.history_manager = ThroughputLimitHistoryManager(lower_bound=0, num_points=1000)  # Set a lower bound for throughput (e.g., 10 points/sec)
    start_time = time.time()
    algorithm.start()
    end_time = time.time()
    result = eval_against_truth(
        initial_text_assignments=algorithm.history_manager.initial_text_assignments,
        final_text_assignments=algorithm.history_manager.get_final_assignments(),
        true_assignments=algorithm.history_manager.true_assignments
    )
    throughput = len(algorithm.history_manager.true_assignments) / (end_time - start_time)
    return {"-nmi":-1 * result.get("v_measure_final", 0.0), 
            "-throughput":-np.log(throughput)}  # Negate because SMAC minimizes by default
    

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
        objectives=["-nmi", "-throughput"],  
        instances=selected_datasets,
        instance_features={name: [i] for i, name in enumerate(selected_datasets)},  # Map instance names to indices
        deterministic=True,
        walltime_limit=60 * 60 * 24 * 7,
        n_trials=250,
        seed=parsed_args.seed,
        crash_cost = [2.0, 2.0],  # Cost for crashing runs (e.g., due to low throughput)
        output_directory=os.path.join("smac3_output", f"loo_{parsed_args.loo_dataset}_seed_{parsed_args.seed}")
    )

    # Use SMAC to find the best configuration/hyperparameters
    smac = MultiObjectiveFacade(scenario, train)  # Use multi-objective optimization if needed
    incumbents = smac.optimize()

    print("Best found configuration:", incumbents)
    
    #Store the best configuration in a JSON file for later use
    best_config_path = os.path.join("smac3_results", f"best_config_loo_{parsed_args.loo_dataset}_seed_{parsed_args.seed}.json")
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump([dict(i.items()) for i in incumbents], f, indent=4)