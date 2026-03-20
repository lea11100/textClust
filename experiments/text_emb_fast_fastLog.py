# Compare the performance of textClust, embClust, and fastClust

import time
import numpy as np
import pandas as pd

from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.algorithms.presets import get_textClust, get_embClust, get_fastClust, get_fastLogClust


def test_algorithm(algorithm: textClust, name: str = "textClust") -> dict:
    algorithm.verbose = False
    print(f"Starting {name}...")
    start_time = time.time()
    algorithm.start()
    end_time = time.time()
    print(f"{name} finished in {end_time - start_time:.2f} seconds.")
    
    flat_result = { 
                   "algorithm": name,
                   "duration": end_time - start_time,
                   "avg_num_macroclusters": np.average([h.get("num_macro_clusters", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_v_meassure": np.average([h.get("v_measure", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_homogeneity": np.average([h.get("homogeneity", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_completeness": np.average([h.get("completeness", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_v_meassure_final": np.average([h.get("v_measure_final", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_homogeneity_final": np.average([h.get("homogeneity_final", 0.0) for h in algorithm.history_manager.eval_history]),
                   "avg_completeness_final": np.average([h.get("completeness_final", 0.0) for h in algorithm.history_manager.eval_history])
                   }
    
    return flat_result

if __name__ == "__main__":
    dataset = "examples/data/Semantic-T_peaked.json"
    data_loader = StaticJsonWrapper(dataset)

    text_clusterer = get_textClust(
        data_loader=data_loader,
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )

    emb_clusterer = get_embClust(
        data_loader=data_loader,
        embedding_model_name='sentence-transformers/paraphrase-MiniLM-L3-v2',
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1,
        device="mps"
    )

    fast_clusterer = get_fastClust(
        data_loader=data_loader,
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )
    
    fastLog_clusterer = get_fastLogClust(
        data_loader=data_loader,
        clean_interval=100,
        cluster_interval=1000,
        log_interval=100,
        decay_rate=0.005,
        sigma=1
    )

    results = []
    
    names = ["fastLogClust", "fastClust"]
    # names = ["textClust", "embClust", "fastLogClust", "fastClust"]
    
    
    for clusterer, name in zip([fastLog_clusterer, fast_clusterer], names):
        result = test_algorithm(clusterer, name=name)
        results.append(result)
        
    df = pd.DataFrame(results)
    print(df)
    
    