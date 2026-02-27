from textClust.data.point import RawPoint
from textClust.history.manager import HistoryManager, Snapshot
from textClust.online.cluster import Microcluster


def test_snapshot_captures_cluster_ids_weights_and_time():
    microclusters = [
        Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1, weight=2.5),
        Microcluster(id=2, tf={"b": 1.0}, current_time=0.0, n=1, weight=3.0),
    ]

    snapshot = Snapshot(microclusters=microclusters, time=12.0)

    assert snapshot.microcluster_ids == [1, 2]
    assert snapshot.microcluster_weights == [2.5, 3.0]
    assert snapshot.time == 12.0


def test_history_manager_initializes_empty_collections():
    manager = HistoryManager()

    assert manager.initial_text_assignments == []
    assert manager.true_assignments == []
    assert manager.snapshots == []
    assert manager.cluster_expiration_time == []
    assert manager.cluster_merge_history == []


def test_update_text_assignment_records_initial_assignment_and_true_assignment_when_present():
    manager = HistoryManager()
    point = RawPoint(text="doc", text_id="t-1", cluster_id=7)

    manager.update_text_assignment(current_time=5.0, point=point, cluster_id=3)

    assert manager.initial_text_assignments == [(5.0, "t-1", 3)]
    assert manager.true_assignments == [("t-1", 7)]


def test_update_text_assignment_skips_true_assignment_when_missing_cluster_id():
    manager = HistoryManager()
    point = RawPoint(text="doc", text_id="t-2", cluster_id=None)

    manager.update_text_assignment(current_time=9.0, point=point, cluster_id=4)

    assert manager.initial_text_assignments == [(9.0, "t-2", 4)]
    assert manager.true_assignments == []


def test_snapshot_method_appends_snapshot_object():
    manager = HistoryManager()
    microclusters = [Microcluster(id=10, tf={"k": 1.0}, current_time=0.0, n=1, weight=1.2)]

    manager.snapshot(current_time=15.0, microclusters=microclusters)

    assert len(manager.snapshots) == 1
    assert isinstance(manager.snapshots[0], Snapshot)
    assert manager.snapshots[0].microcluster_ids == [10]
    assert manager.snapshots[0].microcluster_weights == [1.2]
    assert manager.snapshots[0].time == 15.0


def test_snapshot_stores_weight_values_not_live_references():
    manager = HistoryManager()
    microcluster = Microcluster(id=9, tf={"x": 1.0}, current_time=0.0, n=1, weight=2.0)

    manager.snapshot(current_time=2.0, microclusters=[microcluster])
    microcluster.weight = 9.5

    assert manager.snapshots[0].microcluster_weights == [2.0]


def test_update_history_records_expired_and_merged_info():
    manager = HistoryManager()

    manager.update_history(
        current_time=20.0,
        expired_info=[1, 2],
        merged_info=[(3, [10, 12]), (4, [11])],
    )

    assert manager.cluster_expiration_time == [(20.0, 1), (20.0, 2)]
    assert manager.cluster_merge_history == [(20.0, 3, [10, 12]), (20.0, 4, [11])]


def test_update_history_with_empty_inputs_makes_no_changes():
    manager = HistoryManager()

    manager.update_history(current_time=30.0, expired_info=[], merged_info=[])

    assert manager.cluster_expiration_time == []
    assert manager.cluster_merge_history == []
