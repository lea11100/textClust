"""Optional end-to-end checks using external Arabic and Semantic raw datasets."""

import os
from pathlib import Path

import pytest

from textClust.algorithms.presets import get_embClust, get_textClust, get_fastLogClust
from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.preprocessing.preprocessor import SimulatedEmbeddingPreprocessor, TFIDFPreprocessorWithTranslation
from textClust.utils.eval import eval_against_truth


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/raw_data"
EMBEDDING_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/embedded"

ARABIC_7_CONFIG = {
    "adaptive_threshold": 1,
    "clean_interval": 212,
    "cluster_interval": 829,
    "decay_rate": 0.0035806669979,
    "min_num_mc": 2,
    "use_heuristic": 1,
    "sigma": 0.1794462037772,
    "threshold_momentum": 0.9966134792097,
    "time_gap": 868.874315642948,
}
ARABIC_7_TARGET = {
    "adj_mutual_info": {"target": 0.644886, "lower_bound": 0.6, "upper_bound": 0.7}
}
SEMANTIC_10T_CONFIG = {
    "adaptive_threshold": 1,
    "clean_interval": 343,
    "cluster_interval": 503,
    "decay_rate": 5.53955969e-05,
    "min_num_mc": 16,
    "model": "minishlab_32M",
    "use_heuristic": 1,
    "sigma": 1.1415051047858,
    "threshold_momentum": 0.6698339064938,
    "time_gap": 480.1905888867767
}   
SEMANTIC_10T_TARGET = {
    "adj_mutual_info_final": {"target": 0.839724, "lower_bound": 0.8, "upper_bound": 0.9}
}


def generic_test_clust(data_wrapper: StaticJsonWrapper | SocialEDWrapper, config, target):
    dataset = Path(data_wrapper.file_path)
    if not dataset.is_file():
        pytest.skip(f"raw dataset not available: {dataset}")

    pipeline = get_embClust(
        data_loader=data_wrapper,
        log_interval=1000,
        **config
    )
    embedding_model_name = config.get("model", "minishlab_32M")
    pipeline.preprocessor = SimulatedEmbeddingPreprocessor(embedding_model_name, dataset=Path(data_wrapper.file_path).stem, folder=str(EMBEDDING_FOLDER))
    pipeline.verbose = False  # Avoid interactive plots; still run clustering and evaluation.

    pipeline.start()
    
    result = eval_against_truth(
            initial_text_assignments=pipeline.history_manager.initial_text_assignments,
            final_text_assignments=pipeline.history_manager.get_final_assignments(),
            true_assignments=pipeline.history_manager.true_assignments
        )

    for metric in target:
        assert metric in result, f"Missing metric {metric} in result"
        assert target[metric]["lower_bound"] <= result[metric] <= target[metric]["upper_bound"], f"{metric} {result[metric]} not within target range {target[metric]['lower_bound']} - {target[metric]['upper_bound']}"

@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_embClust_Arabic_7():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Arabic_7.npy")))
    data_wrapper = SocialEDWrapper(str(dataset))
    generic_test_clust(data_wrapper, ARABIC_7_CONFIG, ARABIC_7_TARGET)
    
@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_embClust_Semantic_10T():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Semantic10-T.json")))
    data_wrapper = StaticJsonWrapper(str(dataset))
    generic_test_clust(data_wrapper, SEMANTIC_10T_CONFIG, SEMANTIC_10T_TARGET)