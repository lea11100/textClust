import numpy as np

from textClust.data.point import Point, RawPoint, point_from_raw_point
from textClust.online.cluster import Microcluster


def test_raw_point_constructor_stores_fields_and_extra_kwargs():
    raw_point = RawPoint(
        text="hello world",
        timestamp=42,
        text_id="t-1",
        cluster_id=7,
        source="twitter",
        language="en",
    )

    assert raw_point.text == "hello world"
    assert raw_point.timestamp == 42
    assert raw_point.text_id == "t-1"
    assert raw_point.cluster_id == 7
    assert raw_point.extra == {"source": "twitter", "language": "en"}


def test_raw_point_to_microcluster_defaults_timestamp_to_zero_when_missing():
    raw_point = RawPoint(text="hello")

    microcluster = raw_point.to_microcluster()

    assert isinstance(microcluster, Microcluster)
    assert microcluster.id is None
    assert microcluster.embedding == {}
    assert microcluster.last_update_time == 0
    assert microcluster.n == 1
    assert microcluster.weight == 1.0


def test_raw_point_to_microcluster_uses_timestamp_when_present():
    raw_point = RawPoint(text="hello", timestamp=123)

    microcluster = raw_point.to_microcluster()

    assert microcluster.last_update_time == 123


def test_point_constructor_stores_tokens_embedding_and_extra_kwargs():
    embedding = np.array([1.0, 2.0, 3.0])
    point = Point(
        text="hello world",
        processed_text_tokens=["hello", "world"],
        embedding=embedding,
        timestamp=10,
        text_id="id-1",
        cluster_id=3,
        source="news",
    )

    assert point.text == "hello world"
    assert point.extra["processed_text_tokens"] == ["hello", "world"]
    assert point.embedding is embedding
    assert point.timestamp == 10
    assert point.text_id == "id-1"
    assert point.cluster_id == 3
    assert point.extra == {
        "processed_text_tokens": ["hello", "world"],
        "source": "news",
    }


def test_point_to_microcluster_passes_embedding():
    embedding = np.array([0.1, 0.2])
    point = Point(
        text="sample",
        processed_text_tokens=["a", "b", "a", "c", "b", "a"],
        embedding=embedding,
        timestamp=88,
    )

    microcluster = point.to_microcluster()

    assert isinstance(microcluster, Microcluster)
    assert microcluster.embedding is embedding
    assert microcluster.last_update_time == 88
    assert microcluster.n == 1
    assert microcluster.weight == 1.0


def test_point_to_microcluster_uses_empty_embedding_and_zero_timestamp_when_missing_values():
    point = Point(text="sample", processed_text_tokens=None, timestamp=None)

    microcluster = point.to_microcluster()

    assert microcluster.embedding == {}
    assert microcluster.last_update_time == 0


def test_point_from_raw_point_copies_fields_and_uses_supplied_processing_data():
    raw_point = RawPoint(
        text="original",
        timestamp=55,
        text_id="rid",
        cluster_id=9,
        source="raw-source",
    )
    tokens = ["token1", "token2"]
    embedding = np.array([9.0, 8.0])

    point = point_from_raw_point(
        raw_point,
        processed_text_tokens=tokens,
        embedding=embedding,
        pipeline="default",
    )

    assert isinstance(point, Point)
    assert point.text == "original"
    assert point.timestamp == 55
    assert point.text_id == "rid"
    assert point.cluster_id == 9
    assert point.extra["processed_text_tokens"] == tokens
    assert point.embedding is embedding
    assert point.extra == {"processed_text_tokens": tokens, "pipeline": "default"}
