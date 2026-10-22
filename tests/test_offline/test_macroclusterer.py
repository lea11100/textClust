import pytest

from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster
from textClust.offline.clusterer import MacroClusterer, SimpleMacroClusterer
from textClust.online.cluster import Microcluster


class StubMacroClusterer(MacroClusterer):
    pass


def test_macro_clusterer_base_methods_raise_not_implemented():
    clusterer = StubMacroClusterer()
    history_manager = HistoryManager()

    with pytest.raises(NotImplementedError):
        clusterer.cluster([], history_manager, current_time=0.0)

    with pytest.raises(NotImplementedError):
        clusterer.evaluate([], HistoryManager(), current_time=0.0)

    with pytest.raises(NotImplementedError):
        clusterer.display({})


def test_simple_cluster_converts_microclusters_to_macroclusters():
    clusterer = SimpleMacroClusterer()
    history_manager = HistoryManager()
    microclusters = [
        Microcluster(id=1, current_time=0.0, n=2, embedding={"a": 1.0}, weight=1.5),
        Microcluster(id=2, current_time=0.0, n=3, embedding={"b": 2.0}, weight=2.5),
    ]

    macroclusters, merge_history = clusterer.cluster(microclusters, history_manager, current_time=10.0)

    assert len(macroclusters) == 2
    assert merge_history is None
    assert [mc.id for mc in macroclusters] == [1, 2]
    assert [mc.n for mc in macroclusters] == [2, 3]
    assert [mc.weight for mc in macroclusters] == [1.5, 2.5]
    assert len(history_manager.walltime_at_clustering) == 2
    assert all(isinstance(timestamp, float) for timestamp in history_manager.walltime_at_clustering)


def test_evaluate_sorts_by_weight_and_computes_walltime_diff_without_ground_truth():
    clusterer = SimpleMacroClusterer()
    history_manager = HistoryManager()
    history_manager.walltime_at_clustering = [100.0, 105.0]

    macroclusters = [
        Macrocluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=1, weight=0.5),
        Macrocluster(id=2, embedding={"b": 1.0}, current_time=0.0, n=1, weight=2.0),
        Macrocluster(id=3, embedding={"c": 1.0}, current_time=0.0, n=1, weight=1.0),
    ]

    result = clusterer.evaluate(macroclusters, history_manager=history_manager, current_time=3.0, top_k=2)

    assert result["time"] == 3.0
    assert result["k"] == 2
    assert [mc.id for mc in result["top_k"]] == [2, 3]
    assert result["walltime_diff"] == 5.0
    assert "homogeneity" not in result


def test_make_final_assignment_applies_cluster_merge_history():
    history_manager = HistoryManager()
    history_manager.initial_text_assignments = [
        (0.0, 101, 10),
        (0.0, 102, 20),
        (0.0, 103, 20),
    ]
    history_manager.cluster_merge_history = [
        (1.0, 20, 10),
    ]

    final_assignments = history_manager.get_final_assignments()

    assert sorted(final_assignments) == [(101, 10), (102, 10), (103, 10)]


def test_evaluate_includes_ground_truth_metrics_when_available():
    clusterer = SimpleMacroClusterer()
    history_manager = HistoryManager()
    history_manager.initial_text_assignments = [
        (0.0, 1, 10),
        (0.0, 2, 10),
        (0.0, 3, 20),
        (0.0, 4, 20),
    ]
    history_manager.true_assignments = [
        (1, 0),
        (2, 0),
        (3, 1),
        (4, 1),
    ]
    history_manager.cluster_merge_history = []

    macroclusters = [
        Macrocluster(id=10, embedding={"x": 1.0}, current_time=0.0, n=2, weight=1.0),
        Macrocluster(id=20, embedding={"y": 1.0}, current_time=0.0, n=2, weight=2.0),
    ]

    result = clusterer.evaluate(macroclusters, history_manager=history_manager, current_time=5.0, top_k=2)

    assert result["homogeneity"] == 1.0
    assert result["completeness"] == 1.0
    assert result["v_measure"] == 1.0
    assert result["homogeneity_final"] == 1.0
    assert result["completeness_final"] == 1.0
    assert result["v_measure_final"] == 1.0


def test_display_prints_macrocluster_summary_and_metrics(capsys):
    clusterer = SimpleMacroClusterer()
    evaluation_result = {
        "k": 1,
        "top_k": [Microcluster(id=3, current_time=0.0, n=4, embedding={"alpha": 2.0, "beta": 1.0}, weight=5.0)],
        "walltime_diff": 7.0,
        "homogeneity": 1.0,
        "completeness": 1.0,
        "v_measure": 1.0,
        "homogeneity_final": 1.0,
        "completeness_final": 1.0,
        "v_measure_final": 1.0,
        "num_macro_clusters": 1,
    }

    clusterer.display(evaluation_result)
    output = capsys.readouterr().out
    assert "Time: 7.0 sec" in output
    assert "Evaluation Metrics:" in output
    assert "Number of Micro-Clusters: 1" in output
