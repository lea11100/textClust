import math

import pytest

from textClust.online.cluster import Microcluster
from textClust.online.clusterer import Microclusterer, ThresholdMicroclusterer, TfIdfThresholdMicroclusterer
from textClust.online.fade import DampedFade
from textClust.online.fade import ExpFade
from textClust.online.fast_log_clusterer import FastLogClusterer, FastLogCosineDistance, FastMicrocluster, FastSumMerger
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
        return mc, False, []


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

    idf = clusterer._get_idf(clusterer.microclusters)

    assert idf["a"] == pytest.approx(1 + math.log(3 / 1))
    assert idf["b"] == pytest.approx(1 + math.log(3 / 2))
    assert idf["c"] == pytest.approx(1 + math.log(3 / 2))


def test_tfidf_update_uses_current_idf():
    class RecordingDistance(StubDistanceMetric):
        def __init__(self):
            super().__init__({(1, 99): 0.8, (2, 99): 0.8, (3, 99): 0.8})
            self.metric_kwargs = None

        def __call__(self, mc1, mc2_list, *args, **kwargs):
            self.metric_kwargs = kwargs
            return super().__call__(mc1, mc2_list, *args, **kwargs)

    metric = RecordingDistance()
    clusterer = TfIdfThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=TrackingMerger(),
        adaptive_threshold=False,
        radius=0.5,
        min_num_mc=0,
    )
    clusterer.microclusters = [
        Microcluster(id=1, embedding={"a": 1.0, "b": 1.0}, current_time=0.0, n=1),
        Microcluster(id=2, embedding={"b": 1.0}, current_time=0.0, n=1),
        Microcluster(id=3, embedding={"c": 1.0}, current_time=0.0, n=1),
    ]

    clusterer._update(
        Microcluster(id=99, embedding={"a": 1.0}, current_time=1.0, n=1), current_time=1.0
    )

    assert metric.metric_kwargs is not None
    assert metric.metric_kwargs["n"] == 3
    assert metric.metric_kwargs["idf"]["a"] == pytest.approx(1 + math.log(3))
    assert metric.metric_kwargs["idf"]["b"] == pytest.approx(1 + math.log(3 / 2))


def test_tfidf_cleanup_recomputes_idf_after_each_merge():
    class RecordingDistance(StubDistanceMetric):
        def __init__(self):
            super().__init__({(1, 2): 0.1, (1, 3): 0.9, (1, 4): 0.9,
                              (3, 4): 0.9})
            self.calls = []

        def __call__(self, mc1, mc2_list, *args, **kwargs):
            self.calls.append((kwargs["n"], kwargs["idf"].copy()))
            return super().__call__(mc1, mc2_list, *args, **kwargs)

    metric = RecordingDistance()
    clusterer = TfIdfThresholdMicroclusterer(
        distance_metric=metric,
        fader=IdentityFader(),
        merger=TrackingMerger(),
        adaptive_threshold=False,
        radius=0.5,
    )
    clusterer.microclusters = [
        Microcluster(id=i, embedding={term: 1.0}, current_time=0.0, n=1)
        for i, term in enumerate(("a", "b", "c", "d"), start=1)
    ]

    expired, merged = clusterer.clean_up(current_time=1.0)

    assert expired == []
    assert merged == {2: 1}
    assert [mc.id for mc in clusterer.microclusters] == [1, 3, 4]
    assert [n for n, _ in metric.calls] == [4, 3]
    assert metric.calls[0][1]["a"] == pytest.approx(1 + math.log(4))
    assert metric.calls[1][1]["a"] == pytest.approx(1 + math.log(3))
    assert metric.calls[1][1]["b"] == pytest.approx(1 + math.log(3))


def test_delete_expired_removes_terms_reported_by_fader():
    clusterer = ThresholdMicroclusterer(
        distance_metric=StubDistanceMetric({}),
        fader=ExpFade(decay_rate=1, term_fading=True, min_weight=0.5),
        merger=TrackingMerger(),
        adaptive_threshold=False,
        radius=0.5,
    )
    mc = Microcluster(id=1, embedding={"keep": 2.0, "drop": 0.5}, current_time=0, n=1, weight=2.0)
    clusterer.microclusters = [mc]

    assert clusterer._get_expried(current_time=1) == [(mc, False, ["drop"])]
    assert mc.embedding == {"keep": 1.0, "drop": 0.25}
    assert clusterer.microclusters == [mc]
    assert clusterer._delete_expired(current_time=1) == []
    assert mc.embedding == {"keep": 1.0}


def test_fast_log_cleanup_updates_idf_for_expired_terms_and_clusters():
    clusterer = FastLogClusterer(
        distance_metric=FastLogCosineDistance(),
        fader=ExpFade(decay_rate=1, term_fading=True, min_weight=0.5),
        merger=FastSumMerger(),
        adaptive_threshold=False,
        radius=0.5,
    )
    keep = FastMicrocluster(1, {"keep": 2.0, "drop": 0.5}, clusterer.idf_manager, 0, 1, weight=2.0)
    expire = FastMicrocluster(2, {"gone": 0.5}, clusterer.idf_manager, 0, 1, weight=2.0)
    clusterer.microclusters = [keep, expire]
    clusterer.idf_manager.add_new_microcluster(keep)
    clusterer.idf_manager.add_new_microcluster(expire)

    assert clusterer._get_expried(current_time=1) == [
        (keep, False, ["drop"]), (expire, True, ["gone"])
    ]
    assert clusterer.idf_manager.total_docs == 2
    assert "drop" in keep.embedding and "gone" in expire.embedding
    assert clusterer._delete_expired(current_time=1) == [2]
    assert clusterer.microclusters == [keep]
    assert keep.embedding == {"keep": 1.0}
    assert clusterer.idf_manager.total_docs == 1
    assert dict(clusterer.idf_manager.term_doc_freq) == {"keep": 1}
    assert dict(clusterer.idf_manager.term_doc_occurrence) == {"keep": {1}}
