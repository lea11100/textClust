"""Optional end-to-end checks using external Arabic and Semantic raw datasets."""

import os
from pathlib import Path

import pytest
import numpy as np
from warnings import warn

from textClust.algorithms.presets import get_textClust, get_fastLogClust, get_fastSqrtClust
from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessorWithTranslation
from textClust.utils.eval import eval_against_truth


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/raw_data"
TRANSLATIONS_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/translated_no_emojis"

ARABIC_7_CONFIG = {
    "adaptive_threshold": 1,
    "clean_interval": 368,
    "cluster_interval": 4970,
    "decay_rate": 0.0161279667065,
    "min_num_mc": 1,
    "use_heuristic": 0,
    "min_weight": 7.3226573290961,
    "sigma": 0.9323368982806,
    "threshold_momentum": 0.2788516352087
}
ARABIC_7_TARGET = {
    "adj_mutual_info_final": {"target": 0.635844, "lower_bound": 0.6, "upper_bound": 0.7}
}
SEMANTIC_40T_CONFIG = {
    "adaptive_threshold": 1,
    "clean_interval": 477,
    "cluster_interval": 4910,
    "decay_rate": 0.0318037082557,
    "min_num_mc": 2,
    "use_heuristic": 0,
    "min_weight": 6.4706400999323,
    "sigma": 0.8476769876938,
    "threshold_momentum": 0.0930481298885,
}
SEMANTIC_40T_TARGET = {
    "adj_mutual_info_final": {"target": 0.3522710, "lower_bound": 0.3, "upper_bound": 0.4}
}


def generic_test_clust(data_wrapper: StaticJsonWrapper | SocialEDWrapper, config, target, factory):
    dataset = Path(data_wrapper.file_path)
    if not dataset.is_file():
        pytest.skip(f"raw dataset not available: {dataset}")
    translation = TRANSLATIONS_FOLDER / f"{dataset.stem}.npy"
    if not translation.is_file():
        pytest.skip(f"translated dataset not available: {translation}")

    pipeline = factory(
        data_loader=data_wrapper,
        log_interval=1000,
        **config
    )
    pipeline.preprocessor = TFIDFPreprocessorWithTranslation(dataset=Path(data_wrapper.file_path).stem, folder=str(TRANSLATIONS_FOLDER))
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
        if np.isclose(target[metric]["target"], result[metric]):  # Only check closeness if the target is not equal to the result
            warn(f"{metric} {result[metric]} is close to target {target[metric]['target']}, but not within the specified range.")

@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_fastLogClust_Arabic_7():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Arabic_7.npy")))
    data_wrapper = SocialEDWrapper(str(dataset))
    generic_test_clust(data_wrapper, ARABIC_7_CONFIG, ARABIC_7_TARGET, get_fastLogClust)
    
@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_fastLogClust_Semantic_40T():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Semantic40-T.json")))
    data_wrapper = StaticJsonWrapper(str(dataset))
    generic_test_clust(data_wrapper, SEMANTIC_40T_CONFIG, SEMANTIC_40T_TARGET, get_fastLogClust)


@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_textClust_Arabic_7():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Arabic_7.npy")))
    generic_test_clust(SocialEDWrapper(str(dataset)), ARABIC_7_CONFIG, ARABIC_7_TARGET, get_textClust)


@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_textClust_Semantic_40T():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Semantic40-T.json")))
    generic_test_clust(StaticJsonWrapper(str(dataset)), SEMANTIC_40T_CONFIG, SEMANTIC_40T_TARGET, get_textClust)
    

@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_fastSqrtClust_Arabic_7():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Arabic_7.npy")))
    generic_test_clust(SocialEDWrapper(str(dataset)), ARABIC_7_CONFIG, ARABIC_7_TARGET, get_fastSqrtClust)
    
@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
def test_fastSqrtClust_Semantic_40T():
    dataset = Path(os.environ.get("TEXTCLUST_RAW_DATASET", str(DATASET_FOLDER / "Semantic40-T.json")))
    generic_test_clust(StaticJsonWrapper(str(dataset)), SEMANTIC_40T_CONFIG, SEMANTIC_40T_TARGET, get_fastSqrtClust)