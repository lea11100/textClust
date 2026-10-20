import math

import pytest

from textClust.online.cluster import Microcluster
from textClust.online.clusterer import Microclusterer, ThresholdMicroclusterer, TfIdfThresholdMicroclusterer
from textClust.online.fade import DampedFade
from textClust.online.merge import Merger
from textClust.online.metrics import DistanceMetric


class NoopMicroclusterer(Microclusterer):
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs):
        return mc, None

    def get_clusters(self):
        return []

    def clean_up(self, current_time: float, *args, **kwargs):
        return [], {}


class StubDistanceMetric(DistanceMetric):
    def __init__(self, distances):
        self.distances = distances

    def __call__(self, mc1, mc2_list, *args, **kwargs):
        return [self.distances[tuple(sorted((mc1.id, mc2.id)))] for mc2 in mc2_list]


class IdentityFader(DampedFade):
    def __call__(self, mc, current_time, *args, **kwargs):
        mc.last_update_time = current_time
        return mc, False


class TrackingMerger(Merger):
    def __init__(self):
        self.calls = []

    def __call__(self, mc1, mc2, current_time, *args, **kwargs):
        self.calls.append((mc1.id, mc2.id, current_time))
        assert isinstance(mc1.embedding, dict)
        assert isinstance(mc2.embedding, dict)
        embedding = dict(mc1.embedding)
        for term, value in mc2.embedding.items():
            embedding[term] = embedding.get(term, 0.0) + value
        return Microcluster(
            id=mc1.id,
            embedding=embedding,
            current_time=current_time,
            n=mc1.n + mc2.n,
            weight=mc1.weight + mc2.weight,
        )


def test_microclusterer_update_assigns_new_cluster_id_when_update_returns_none():
    clusterer = NoopMicroclusterer()
    incoming = Microcluster(id=-1, embedding={"x": 1.0}, current_time=0.0, n=1)

    assigned_id = clusterer.update(incoming, current_time=1.0)

    assert assigned_id == 1
    assert incoming.id == 1


def test_threshold_update_merges_into_closest_cluster_below_radius():
    metric = StubDistanceMetric({(10, 99): 0.1, (20, 99): 0.8})
    merger = TrackingMerger()
    clusterer = ThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=merger,
        adaptive_threshold=False,
        radius=0.5,
        min_num_mc=0,
    )
    clusterer.microclusters = [
        Microcluster(id=10, embedding={"a": 1.0}, current_time=0.0, n=1),
        Microcluster(id=20, embedding={"b": 1.0}, current_time=0.0, n=1),
    ]

    result, assigned_id = clusterer._update(
        Microcluster(id=99, embedding={"a": 2.0}, current_time=0.0, n=1), current_time=5.0
    )

    assert assigned_id == 10
    assert result.id == 10
    assert merger.calls == [(10, 99, 5.0)]
    assert clusterer.num_merged_obs == 1


def test_threshold_update_appends_outlier_above_radius():
    clusterer = ThresholdMicroclusterer(
        distance_metric=StubDistanceMetric({(10, 99): 0.8}),
        fader=IdentityFader(),
        merger=TrackingMerger(),
        adaptive_threshold=False,
        radius=0.5,
        min_num_mc=0,
    )
    clusterer.microclusters = [Microcluster(id=10, embedding={"a": 1.0}, current_time=0.0, n=1)]
    incoming = Microcluster(id=99, embedding={"c": 1.0}, current_time=0.0, n=1)

    result, assigned_id = clusterer._update(incoming, current_time=5.0)

    assert result is incoming
    assert assigned_id is None
    assert clusterer.microclusters[-1] is incoming
    assert clusterer.outlier_dist_sum == pytest.approx(0.8)


def test_tfidf_threshold_clusterer_computes_idf():
    clusterer = TfIdfThresholdMicroclusterer(
        distance_metric=StubDistanceMetric({}),
        fader=IdentityFader(),
        merger=TrackingMerger(),
        adaptive_threshold=False,
        radius=0.5,
    )
    clusterer.microclusters = [
        Microcluster(id=1, embedding={"a": 1.0, "b": 1.0}, current_time=0.0, n=1),
        Microcluster(id=2, embedding={"b": 2.0, "c": 1.0}, current_time=0.0, n=1),
        Microcluster(id=3, embedding={"c": 3.0}, current_time=0.0, n=1),
    ]

    idf = clusterer._get_idf()

    assert idf["a"] == pytest.approx(1 + math.log(3 / 1))
    assert idf["b"] == pytest.approx(1 + math.log(3 / 2))
    assert idf["c"] == pytest.approx(1 + math.log(3 / 2))
