import numpy as np
import pytest

from textClust.online.cluster import Microcluster
from textClust.online.metrics import DistanceMetric, TfidfCosineDistance


def test_distance_metric_base_defaults_and_call_raises():
    metric = DistanceMetric()

    assert metric.requires_idf is False

    mc1 = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, tf={"a": 1.0}, current_time=0.0, n=1)

    with pytest.raises(NotImplementedError):
        metric(mc1, mc2)


def test_tfidf_cosine_distance_sets_requires_idf_true():
    metric = TfidfCosineDistance()

    assert metric.requires_idf is True


def test_tfidf_cosine_distance_returns_zero_when_no_term_overlap():
    metric = TfidfCosineDistance()
    mc1 = Microcluster(id=1, tf={"a": 1.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, tf={"b": 2.0}, current_time=0.0, n=1)

    distance = metric(mc1, mc2, idf={"a": 1.2, "b": 1.4})

    assert distance == 1.0


def test_tfidf_cosine_distance_without_idf_matches_expected_value():
    metric = TfidfCosineDistance()
    mc1 = Microcluster(id=1, tf={"a": 1.0, "b": 2.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, tf={"a": 3.0, "c": 4.0}, current_time=0.0, n=1)

    distance = metric(mc1, mc2, idf={})

    inner_product = 1.0 * 3.0
    norm1 = np.linalg.norm(np.array([1.0, 2.0]))
    norm2 = np.linalg.norm(np.array([3.0, 4.0]))
    expected_distance = 1 - (inner_product / (norm1 * norm2))

    assert distance == pytest.approx(expected_distance)


def test_tfidf_cosine_distance_with_idf_uses_weighted_inner_product():
    metric = TfidfCosineDistance()
    mc1 = Microcluster(id=1, tf={"a": 1.0, "b": 2.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, tf={"a": 3.0, "c": 4.0}, current_time=0.0, n=1)
    idf = {"a": 2.0, "b": 10.0, "c": 10.0}

    distance = metric(mc1, mc2, idf=idf)

    inner_product = (idf["a"] ** 2) * 1.0 * 3.0
    norm1 = np.linalg.norm(np.array([idf["a"] * 1.0, idf["b"] * 2.0]))
    norm2 = np.linalg.norm(np.array([idf["a"] * 3.0, idf["c"] * 4.0]))
    expected_distance = 1 - (inner_product / (norm1 * norm2))

    assert distance == pytest.approx(expected_distance)


def test_make_tfidf_vectors_uses_idf_and_defaults_to_one_for_missing_terms():
    metric = TfidfCosineDistance()

    tfidf1, tfidf2 = metric._make_tfidf_vectors(
        tf1={"a": 1.0, "b": 2.0},
        tf2={"c": 3.0, "d": 4.0},
        idf={"a": 2.0, "c": 5.0},
    )

    assert np.array_equal(tfidf1, np.array([2.0, 2.0]))
    assert np.array_equal(tfidf2, np.array([15.0, 4.0]))


def test_make_tfidf_vectors_without_idf_returns_tf_values():
    metric = TfidfCosineDistance()

    tfidf1, tfidf2 = metric._make_tfidf_vectors(
        tf1={"a": 1.0, "b": 2.0},
        tf2={"c": 3.0},
        idf={},
    )

    assert np.array_equal(tfidf1, np.array([1.0, 2.0]))
    assert np.array_equal(tfidf2, np.array([3.0]))
