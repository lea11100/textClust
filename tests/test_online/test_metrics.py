import numpy as np
import pytest

from textClust.online.cluster import Microcluster
from textClust.online.metrics import CosineDistance, DistanceMetric


def test_distance_metric_base_defaults_and_call_raises():
    metric = DistanceMetric()

    mc1 = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, embedding={"a": 1.0}, current_time=0.0, n=1)

    with pytest.raises(NotImplementedError):
        metric(mc1, mc2)


def test_cosine_distance_returns_one_when_no_term_overlap():
    metric = CosineDistance()
    mc1 = Microcluster(id=1, embedding={"a": 1.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, embedding={"b": 2.0}, current_time=0.0, n=1)

    distance = metric(mc1, [mc2], idf={"a": 1.2, "b": 1.4}, n=2)

    assert distance == [1.0]


def test_tfidf_cosine_distance_without_idf_matches_expected_value():
    metric = CosineDistance()
    mc1 = Microcluster(id=1, embedding={"a": 1.0, "b": 2.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, embedding={"a": 3.0, "c": 4.0}, current_time=0.0, n=1)

    distance = metric(mc1, [mc2], idf={}, n=2)[0]

    inner_product = 1.0 * 3.0
    norm1 = np.linalg.norm(np.array([1.0, 2.0]))
    norm2 = np.linalg.norm(np.array([3.0, 4.0]))
    expected_distance = 1 - (inner_product / (norm1 * norm2))

    assert distance == pytest.approx(expected_distance)


def test_tfidf_cosine_distance_with_idf_uses_weighted_inner_product():
    metric = CosineDistance()
    mc1 = Microcluster(id=1, embedding={"a": 1.0, "b": 2.0}, current_time=0.0, n=1)
    mc2 = Microcluster(id=2, embedding={"a": 3.0, "c": 4.0}, current_time=0.0, n=1)
    idf = {"a": 2.0, "b": 10.0, "c": 10.0}

    distance = metric(mc1, [mc2], idf=idf, n=2)[0]

    inner_product = (idf["a"] ** 2) * 1.0 * 3.0
    norm1 = np.linalg.norm(np.array([idf["a"] * 1.0, idf["b"] * 2.0]))
    norm2 = np.linalg.norm(np.array([idf["a"] * 3.0, idf["c"] * 4.0]))
    expected_distance = 1 - (inner_product / (norm1 * norm2))

    assert distance == pytest.approx(expected_distance)


