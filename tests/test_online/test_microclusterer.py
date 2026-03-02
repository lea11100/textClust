import math

import pytest

from textClust.online.cluster import Microcluster
from textClust.online.clusterer import AdaptiveThresholdMicroclusterer, Microclusterer
from textClust.online.fade import DampedFade
from textClust.online.merge import Merger
from textClust.online.metrics import DistanceMetric


class NoopMicroclusterer(Microclusterer):
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs):
        return None

    def get_clusters(self):
        return []

    def clean_up(self, current_time: float, *args, **kwargs):
        return [], {}


class StubDistanceMetric(DistanceMetric):
    def __init__(self, distances: dict[tuple[int, int], float], requires_idf: bool = False):
        super().__init__()
        self.distances = distances
        self.requires_idf = requires_idf
        self.captured_idf = []

    def __call__(self, mc1: Microcluster, mc2: Microcluster, idf=None, *args, **kwargs) -> float:
        self.captured_idf.append(idf)
        key = tuple(sorted((mc1.id, mc2.id)))
        return self.distances[key]


class IdentityFader(DampedFade):
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs):
        mc.last_update_time = current_time
        return mc, False


class SelectiveExpireFader(DampedFade):
    def __init__(self, expired_ids: set[int]):
        self.expired_ids = expired_ids

    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs):
        mc.last_update_time = current_time
        return mc, mc.id in self.expired_ids


class TrackingMerger(Merger):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        self.calls.append((mc1.id, mc2.id, current_time))
        merged_tf = dict(mc1.tf)
        for term, value in mc2.tf.items():
            merged_tf[term] = merged_tf.get(term, 0.0) + value
        return Microcluster(
            id=mc1.id,
            tf=merged_tf,
            current_time=current_time,
            n=mc1.n + mc2.n,
            weight=mc1.weight + mc2.weight,
        )


def test_microclusterer_update_assigns_new_cluster_id_when_update_returns_none():
    clusterer = NoopMicroclusterer()
    incoming = Microcluster(id=-1, tf={"x": 1.0}, current_time=0.0, n=1)

    assigned_id = clusterer.update(incoming, current_time=1.0)

    assert assigned_id == 1
    assert incoming.id == 1


def test_adaptive_update_merges_into_closest_cluster_when_below_threshold():
    metric = StubDistanceMetric(distances={(10, 99): 1.0, (20, 99): 3.0})
    merger = TrackingMerger()
    clusterer = AdaptiveThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=merger,
        sigma=1.0,
    )
    clusterer.microclusters = [
        Microcluster(id=10, tf={"a": 1.0}, current_time=0.0, n=1),
        Microcluster(id=20, tf={"b": 1.0}, current_time=0.0, n=1),
    ]
    incoming = Microcluster(id=99, tf={"a": 2.0}, current_time=0.0, n=1)

    result = clusterer._update(incoming, current_time=5.0)

    assert result == 10
    assert merger.calls == [(10, 99, 5.0)]
    assert clusterer.num_merged_obs == 1
    assert clusterer.dist_sum == pytest.approx(1.0)
    assert clusterer.microclusters[0].id == 10
    assert clusterer.microclusters[0].n == 2


def test_adaptive_update_appends_outlier_when_not_below_threshold():
    metric = StubDistanceMetric(distances={(10, 99): 3.0, (20, 99): 3.0})
    clusterer = AdaptiveThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=TrackingMerger(),
        sigma=1.0,
    )
    clusterer.microclusters = [
        Microcluster(id=10, tf={"a": 1.0}, current_time=0.0, n=1),
        Microcluster(id=20, tf={"b": 1.0}, current_time=0.0, n=1),
    ]
    incoming = Microcluster(id=99, tf={"c": 1.0}, current_time=0.0, n=1)

    result = clusterer._update(incoming, current_time=5.0)

    assert result is None
    assert clusterer.microclusters[-1] is incoming
    assert clusterer.outlier_dist_sum == pytest.approx(3.0)
    assert clusterer.outlier_dist_squaresum == pytest.approx(9.0)


def test_adaptive_update_computes_and_passes_idf_when_metric_requires_it():
    metric = StubDistanceMetric(distances={(10, 99): 1.0, (20, 99): 2.0}, requires_idf=True)
    clusterer = AdaptiveThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=TrackingMerger(),
        sigma=1.0,
    )
    clusterer.microclusters = [
        Microcluster(id=10, tf={"shared": 1.0, "a": 1.0}, current_time=0.0, n=1),
        Microcluster(id=20, tf={"shared": 2.0, "b": 1.0}, current_time=0.0, n=1),
    ]

    clusterer._update(Microcluster(id=99, tf={"a": 1.0}, current_time=0.0, n=1), current_time=1.0)

    assert len(metric.captured_idf) == 2
    assert metric.captured_idf[0]["shared"] == pytest.approx(1.0)
    assert metric.captured_idf[0]["a"] == pytest.approx(1 + math.log(2 / 1))
    assert metric.captured_idf[0]["b"] == pytest.approx(1 + math.log(2 / 1))


def test_clean_up_removes_expired_and_returns_merge_history():
    metric = StubDistanceMetric(distances={(1, 2): 0.5})
    merger = TrackingMerger()
    clusterer = AdaptiveThresholdMicroclusterer(
        distance_metric=metric,
        fader=SelectiveExpireFader(expired_ids={3}),
        merger=merger,
        sigma=1.0,
    )
    clusterer.microclusters = [
        Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1),
        Microcluster(id=2, tf={"b": 1.0}, current_time=0.0, n=1),
        Microcluster(id=3, tf={"c": 1.0}, current_time=0.0, n=1),
    ]
    clusterer.num_merged_obs = 1
    clusterer.dist_sum = 1.0

    expired_ids, merge_history = clusterer.clean_up(current_time=10.0)

    assert expired_ids == [3]
    assert merge_history == {2: 1}
    assert [mc.id for mc in clusterer.microclusters] == [1]
    assert merger.calls == [(1, 2, 10.0)]


def test_get_idf_matches_expected_formula():
    clusterer = AdaptiveThresholdMicroclusterer(
        distance_metric=StubDistanceMetric(distances={}),
        fader=IdentityFader(),
        merger=TrackingMerger(),
        sigma=1.0,
    )
    clusterer.microclusters = [
        Microcluster(id=1, tf={"a": 1.0, "b": 1.0}, current_time=0.0, n=1),
        Microcluster(id=2, tf={"b": 2.0, "c": 1.0}, current_time=0.0, n=1),
        Microcluster(id=3, tf={"c": 3.0}, current_time=0.0, n=1),
    ]

    idf = clusterer._get_idf()

    assert idf["a"] == pytest.approx(1 + math.log(3 / 1))
    assert idf["b"] == pytest.approx(1 + math.log(3 / 2))
    assert idf["c"] == pytest.approx(1 + math.log(3 / 2))
