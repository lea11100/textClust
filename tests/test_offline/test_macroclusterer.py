import datetime

import pytest

from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import MacroClusterer, SimpleTfidfMacroClusterer, MacroCluster
from textClust.online.cluster import Microcluster


class StubMacroClusterer(MacroClusterer):
    pass


def test_macro_clusterer_base_methods_raise_not_implemented():
    clusterer = StubMacroClusterer()

    with pytest.raises(NotImplementedError):
        clusterer.cluster([], current_time=0.0)

    with pytest.raises(NotImplementedError):
        clusterer.evaluate([], HistoryManager(), current_time=0.0)

    with pytest.raises(NotImplementedError):
        clusterer.display({})


def test_simple_tfidf_cluster_converts_microclusters_to_macroclusters():
    clusterer = SimpleTfidfMacroClusterer()
    microclusters = [
        Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=2, embedding=[0.1, 0.2], weight=1.5),
        Microcluster(id=2, tf={"b": 2.0}, current_time=0.0, n=3, embedding=None, weight=2.5),
    ]

    macroclusters = clusterer.cluster(microclusters, current_time=10.0)

    assert len(macroclusters) == 2
    assert [mc.id for mc in macroclusters] == [1, 2]
    assert [mc.n for mc in macroclusters] == [2, 3]
    assert [mc.weight for mc in macroclusters] == [1.5, 2.5]
    assert len(clusterer.walltime_at_clustering) == 1
    assert isinstance(clusterer.walltime_at_clustering[0], float)


def test_evaluate_sorts_by_weight_and_computes_walltime_diff_without_ground_truth():
    clusterer = SimpleTfidfMacroClusterer()
    clusterer.walltime_at_clustering = [
        datetime.datetime(2026, 1, 1, 0, 0, 0),
        datetime.datetime(2026, 1, 1, 0, 0, 5),
    ]

    macroclusters = [
        MacroCluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1, weight=0.5),
        MacroCluster(id=2, tf={"b": 1.0}, current_time=0.0, n=1, weight=2.0),
        MacroCluster(id=3, tf={"c": 1.0}, current_time=0.0, n=1, weight=1.0),
    ]

    result = clusterer.evaluate(macroclusters, history_manager=HistoryManager(), current_time=3.0, top_k=2)

    assert result["time"] == 3.0
    assert result["k"] == 2
    assert [mc.id for mc in result["top_k"]] == [2, 3]
    assert len(result["walltime_diff"]) == 1
    assert "homogeneity" not in result


def test_make_final_assignment_applies_cluster_merge_history():
    clusterer = SimpleTfidfMacroClusterer()
    history_manager = HistoryManager()
    history_manager.initial_text_assignments = [
        (0.0, 101, 10),
        (0.0, 102, 20),
        (0.0, 103, 20),
    ]
    history_manager.cluster_merge_history = [
        (1.0, 20, 10),
    ]

    final_assignments = clusterer._make_final_assignent(history_manager)

    assert sorted(final_assignments) == [(101, 10), (102, 10), (103, 10)]


def test_evaluate_includes_ground_truth_metrics_when_available():
    clusterer = SimpleTfidfMacroClusterer()
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
        MacroCluster(id=10, tf={"x": 1.0}, current_time=0.0, n=2, weight=1.0),
        MacroCluster(id=20, tf={"y": 1.0}, current_time=0.0, n=2, weight=2.0),
    ]

    result = clusterer.evaluate(macroclusters, history_manager=history_manager, current_time=5.0, top_k=2)

    assert result["homogeneity"] == 1.0
    assert result["completeness"] == 1.0
    assert result["v_measure"] == 1.0
    assert result["homogeneity_final"] == 1.0
    assert result["completeness_final"] == 1.0
    assert result["v_measure_final"] == 1.0


def test_display_prints_macrocluster_summary_and_metrics(capsys):
    clusterer = SimpleTfidfMacroClusterer()
    evaluation_result = {
        "k": 1,
        "top_k": [Microcluster(id=3, tf={"alpha": 2.0, "beta": 1.0}, current_time=0.0, n=4, weight=5.0)],
        "walltime_diff": [7.0],
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
    print(output)  # For debugging purposes, can be removed later

    assert "Time: 7.0 sec" in output
    assert "Top 1 Micro-Clusters" in output
    assert "Cluster 3" in output
    assert "alpha: 2.00" in output
    assert "Evaluation Metrics:" in output
    assert "Number of Micro-Clusters: 1" in output
