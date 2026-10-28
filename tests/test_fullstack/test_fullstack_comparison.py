"""Compare the runtime and output of textClust and fastLogClust."""

import os
import signal
import time
from pathlib import Path

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


def _assert_same_partition(reference, candidate):
    """Compare cluster assignments independently of cluster ID labels."""
    assert len(reference) == len(candidate), "Different numbers of assignments"
    reference_to_candidate = {}
    candidate_to_reference = {}
    for index, (reference_id, candidate_id) in enumerate(zip(reference, candidate)):
        assert reference_to_candidate.setdefault(reference_id, candidate_id) == candidate_id, (
            f"Reference cluster {reference_id!r} is split at assignment {index}"
        )
        assert candidate_to_reference.setdefault(candidate_id, reference_id) == reference_id, (
            f"Candidate cluster {candidate_id!r} merges distinct clusters at assignment {index}"
        )


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
            pytest.fail(f"Comparison for {dataset.name} exceeded {COMPARISON_TIMEOUT_SECONDS} seconds", pytrace=False)
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

    start = time.perf_counter()
    pipeline_textClust.start()
    end = time.perf_counter()
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

    start = time.perf_counter()
    pipeline_fastLogClust.start()
    end = time.perf_counter()
    time_taken_fastLogClust = end - start

    result_fastLogtextClust = eval_against_truth(
            initial_text_assignments=pipeline_fastLogClust.history_manager.initial_text_assignments,
            final_text_assignments=pipeline_fastLogClust.history_manager.get_final_assignments(),
            true_assignments=pipeline_fastLogClust.history_manager.true_assignments
        )

    reference_history = pipeline_textClust.history_manager
    candidate_history = pipeline_fastLogClust.history_manager
    reference_initial = reference_history.initial_text_assignments
    candidate_initial = candidate_history.initial_text_assignments
    assert [(t, point_id) for t, point_id, _ in reference_initial] == [
        (t, point_id) for t, point_id, _ in candidate_initial
    ], "The pipelines processed different points or timestamps"
    _assert_same_partition(
        [cluster_id for _, _, cluster_id in reference_initial],
        [cluster_id for _, _, cluster_id in candidate_initial],
    )

    reference_final = reference_history.get_final_assignments()
    candidate_final = candidate_history.get_final_assignments()
    reference_by_point = dict(reference_final)
    candidate_by_point = dict(candidate_final)
    assert len(reference_by_point) == len(reference_final), "Duplicate reference point IDs"
    assert len(candidate_by_point) == len(candidate_final), "Duplicate candidate point IDs"
    assert reference_by_point.keys() == candidate_by_point.keys(), "Different final point IDs"
    _assert_same_partition(
        list(reference_by_point.values()),
        [candidate_by_point[point_id] for point_id in reference_by_point],
    )
    assert result_textClust[target] == pytest.approx(result_fastLogtextClust[target], rel=0, abs=1e-12)
    print(f"textClust: {time_taken_textClust:.3f}s; fastLogClust: {time_taken_fastLogClust:.3f}s")


def test_partition_comparison_ignores_cluster_labels():
    _assert_same_partition([1, 1, 2, 3], [8, 8, 9, 7])


@pytest.mark.parametrize("candidate", [[8, 9, 9], [8, 8, 8]])
def test_partition_comparison_rejects_splits_and_merges(candidate):
    with pytest.raises(AssertionError):
        _assert_same_partition([1, 1, 2], candidate)


def test_dataset_comparison_timeout_fails(tmp_path, monkeypatch):
    dataset = tmp_path / "sample.json"
    dataset.touch()
    (tmp_path / "sample.npy").touch()
    monkeypatch.setitem(globals(), "TRANSLATIONS_FOLDER", tmp_path)

    def timeout(*args, **kwargs):
        raise _ComparisonTimedOut

    monkeypatch.setitem(globals(), "_run_textClust_fastClust_comparison", timeout)
    previous_handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(pytest.fail.Exception, match="exceeded"):
        generic_textClust_fastClust_comparison(StaticJsonWrapper(str(dataset)), {}, "metric")
    assert signal.getsignal(signal.SIGALRM) == previous_handler


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