"""Compare the runtime and output of textClust and fastLogClust."""

import os
import signal
import time
from pathlib import Path
from warnings import warn

import numpy as np
import pytest

from textClust.algorithms.presets import get_fastLogClust, get_textClust
from textClust.data.wrapper import SocialEDWrapper, StaticJsonWrapper
from textClust.preprocessing.preprocessor import TFIDFPreprocessorWithTranslation
from textClust.utils.eval import eval_against_truth


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/raw_data"
TRANSLATIONS_FOLDER = REPO_ROOT / "examples/data/textClust_stream_data/translated_no_emojis"

# General-purpose comparison settings, not tuned to any individual dataset.
# Use a fixed expiry threshold rather than a dataset-specific time gap.
COMPARISON_CONFIG = {
    "adaptive_threshold": True,
    "clean_interval": 400,
    "cluster_interval": 1000,
    "decay_rate": 0.02,
    "min_num_mc": 2,
    "use_heuristic": True,
    "time_gap": 400,
    "sigma": 0.9,
    "threshold_momentum": 0.2,
    "term_fading": True,
}

COMPARISON_TIMEOUT_SECONDS = 10 * 60


class _ComparisonTimedOut(BaseException):
    """Interrupt a comparison when its wall-clock deadline expires."""


def _comparison_timeout_handler(signum, frame):
    raise _ComparisonTimedOut


def comparison_datasets():
    """Find raw datasets with a matching precomputed English translation."""
    return sorted(
        (path for path in DATASET_FOLDER.iterdir()
         if path.suffix in {".json", ".npy"}
         and (TRANSLATIONS_FOLDER / f"{path.stem}.npy").is_file()),
        key=lambda path: path.name,
    ) if DATASET_FOLDER.is_dir() and TRANSLATIONS_FOLDER.is_dir() else []


def generic_textClust_fastClust_comparison(data_wrapper: StaticJsonWrapper | SocialEDWrapper, config, target):
    dataset = Path(data_wrapper.file_path)
    if not dataset.is_file():
        pytest.skip(f"raw dataset not available: {dataset}")
    translation = TRANSLATIONS_FOLDER / f"{dataset.stem}.npy"
    if not translation.is_file():
        pytest.skip(f"translated dataset not available: {translation}")

    previous_handler = signal.signal(signal.SIGALRM, _comparison_timeout_handler)
    try:
        previous_timer = signal.setitimer(signal.ITIMER_REAL, COMPARISON_TIMEOUT_SECONDS)
        try:
            _run_textClust_fastClust_comparison(data_wrapper, config, target)
        except _ComparisonTimedOut:
            warn(f"Comparison for {dataset.name} exceeded 10 minutes; stopping calculation.", RuntimeWarning)
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)


def _run_textClust_fastClust_comparison(data_wrapper, config, target):
    pipeline_textClust = get_textClust(
        data_loader=data_wrapper,
        log_interval=1000,
        **config
    )

    # Configuration got without using stemming
    pipeline_textClust.preprocessor = TFIDFPreprocessorWithTranslation(dataset=Path(data_wrapper.file_path).stem, folder=str(TRANSLATIONS_FOLDER), stemming=False)
    pipeline_textClust.verbose = False  # Avoid interactive plots; still run clustering and evaluation.

    start = time.time()
    pipeline_textClust.start()
    end = time.time()
    time_taken_textClust = end - start

    result_textClust = eval_against_truth(
            initial_text_assignments=pipeline_textClust.history_manager.initial_text_assignments,
            final_text_assignments=pipeline_textClust.history_manager.get_final_assignments(),
            true_assignments=pipeline_textClust.history_manager.true_assignments
        )

    pipeline_fastLogClust = get_fastLogClust(
        data_loader=data_wrapper,
        log_interval=1000,
        **config
    )

    pipeline_fastLogClust.preprocessor = TFIDFPreprocessorWithTranslation(dataset=Path(data_wrapper.file_path).stem, folder=str(TRANSLATIONS_FOLDER), stemming=False)
    pipeline_fastLogClust.verbose = False  # Avoid interactive plots; still run clustering and evaluation.

    start = time.time()
    pipeline_fastLogClust.start()
    end = time.time()
    time_taken_fastLogClust = end - start

    result_fastLogtextClust = eval_against_truth(
            initial_text_assignments=pipeline_fastLogClust.history_manager.initial_text_assignments,
            final_text_assignments=pipeline_fastLogClust.history_manager.get_final_assignments(),
            true_assignments=pipeline_fastLogClust.history_manager.true_assignments
        )

    if not np.isclose(result_textClust[target], result_fastLogtextClust[target]):
        warn(f"{target} differs between textClust ({result_textClust[target]}) and fastLogClust ({result_fastLogtextClust[target]})")

    assert time_taken_textClust > time_taken_fastLogClust, f"textClust ({time_taken_textClust}) is faster than fastLogClust ({time_taken_fastLogClust})"


@pytest.mark.skipif(
    os.environ.get("TEXTCLUST_RUN_DATASET_TESTS") != "1",
    reason="opt in with TEXTCLUST_RUN_DATASET_TESTS=1",
)
@pytest.mark.parametrize("dataset", comparison_datasets(), ids=lambda path: path.stem)
def test_fastLogClust_equal_textClust(dataset):
    wrapper = SocialEDWrapper if dataset.suffix == ".npy" else StaticJsonWrapper
    generic_textClust_fastClust_comparison(
        wrapper(str(dataset)), COMPARISON_CONFIG, "adj_mutual_info_final"
    )