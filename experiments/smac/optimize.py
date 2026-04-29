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

datasets = [
    "examples/data/Semantic10-T.json",
    "examples/data/Semantic5-T.json",
    "examples/data/NT.json",
    "examples/data/NTS.json",
    "examples/data/NTSO.json",
    "examples/data/SO-T.json",
    "examples/data/News-T.json",
    "examples/data/Trends-T.json",
    "examples/data/Tweets-T.json",
]

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
    
    

    
if __name__ == "__main__":
    with open(f'experiments/smac/params.pcs', 'r') as f:
        configuration = pcs.read(f)

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configuration, 
        instances=datasets, instance_features={name:[i] for i, name in enumerate(datasets)},  # Map instance names to indices
        deterministic=True, walltime_limit=12 * 60 * 60)

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()

    print("Best found configuration:", incumbent)