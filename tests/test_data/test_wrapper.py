import pytest

from textClust.data.point import RawPoint
from textClust.data.wrapper import StaticJsonWrapper

def test_static_json_wrapper_yields_points_and_none():
    dataset = [
        {"tweets": "hello", "timestamp": 123, "tweetID": "u1", "clusterNo": 7},
        {"tweets": "world", "timestamp": 456, "tweetID": "u2", "clusterNo": 9},
    ]
    wrapper = StaticJsonWrapper(dataset)

    items = list(wrapper.load())

    assert len(items) == 3
    assert isinstance(items[0], RawPoint)
    assert isinstance(items[1], RawPoint)
    assert items[2] is None
    assert items[0].text == "hello"
    assert items[0].timestamp == 123
    assert items[0].user == "u1"
    assert items[0].cluster_id == 7
    assert items[1].text == "world"
    assert items[1].timestamp == 456
    assert items[1].user == "u2"
    assert items[1].cluster_id == 9


def test_static_json_wrapper_handles_missing_fields():
    dataset = [
        {"tweets": "only text"},
        {},
    ]
    wrapper = StaticJsonWrapper(dataset)

    items = list(wrapper.load())

    assert len(items) == 3
    assert isinstance(items[0], RawPoint)
    assert isinstance(items[1], RawPoint)
    assert items[0].text == "only text"
    assert items[0].timestamp is None
    assert items[0].user is None
    assert items[0].cluster_id is None
    assert items[1].text is None
    assert items[1].timestamp is None
    assert items[1].user is None
    assert items[1].cluster_id is None
    assert items[2] is None


def test_static_json_wrapper_retains_dataset_reference():
    dataset = []
    wrapper = StaticJsonWrapper(dataset)

    assert wrapper.dataset is dataset


def test_static_json_wrapper_load_is_repeatable():
    dataset = [
        {"tweets": "a", "timestamp": 1, "tweetID": "u", "clusterNo": 2},
    ]
    wrapper = StaticJsonWrapper(dataset)

    first = list(wrapper.load())
    second = list(wrapper.load())

    assert len(first) == 2
    assert len(second) == 2
    assert isinstance(first[0], RawPoint)
    assert isinstance(second[0], RawPoint)
    assert first[0].text == "a"
    assert second[0].text == "a"
    assert first[1] is None
    assert second[1] is None