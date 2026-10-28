from textClust.data.point import RawPoint
from textClust.data.wrapper import SocialEDWrapper, StaticJsonWrapper
import json
import numpy as np
import pytest

def write_json_lines(file_path, items):
    with open(file_path, "w") as file:
        for item in items:
            file.write(json.dumps(item))
            file.write("\n")


def test_static_json_wrapper_yields_points_and_none(tmp_path):
    dataset = [
        {"tweets": "hello", "timestamp": 123, "tweetID": "u1", "clusterNo": 7},
        {"tweets": "world", "timestamp": 456, "tweetID": "u2", "clusterNo": 9},
    ]
    file_path = tmp_path / "data.jsonl"
    write_json_lines(file_path, dataset)
    wrapper = StaticJsonWrapper(str(file_path))

    items = list(wrapper.load())

    assert len(items) == 3
    assert isinstance(items[0], RawPoint)
    assert isinstance(items[1], RawPoint)
    assert items[2] is None
    assert items[0].text == "hello"
    assert items[0].timestamp == 123
    assert items[0].text_id == "u1"
    assert items[0].cluster_id == 7
    assert items[1].text == "world"
    assert items[1].timestamp == 456
    assert items[1].text_id == "u2"
    assert items[1].cluster_id == 9


def test_static_json_wrapper_handles_missing_fields(tmp_path):
    dataset = [
        {"tweets": "only text"},
        {},
    ]
    file_path = tmp_path / "data.jsonl"
    write_json_lines(file_path, dataset)
    wrapper = StaticJsonWrapper(str(file_path))

    items = list(wrapper.load())

    assert len(items) == 3
    assert isinstance(items[0], RawPoint)
    assert isinstance(items[1], RawPoint)
    assert items[0].text == "only text"
    assert items[0].timestamp is None
    assert items[0].text_id is None
    assert items[0].cluster_id is None
    assert items[1].text is None
    assert items[1].timestamp is None
    assert items[1].text_id is None
    assert items[1].cluster_id is None
    assert items[2] is None


@pytest.mark.parametrize(
    ("record", "expected_timestamp"),
    [
        ({"timestamp": 123}, 123),
        ({"ts": 456}, 456),
        ({"ts": 0, "timestamp": 123}, 0),
        ({"ts": None, "timestamp": 123}, 123),
    ],
)
def test_static_json_wrapper_supports_timestamp_fields(tmp_path, record, expected_timestamp):
    file_path = tmp_path / "data.jsonl"
    write_json_lines(file_path, [record])

    point = next(StaticJsonWrapper(str(file_path)).load())

    assert isinstance(point, RawPoint)
    assert point.timestamp == expected_timestamp


def test_static_json_wrapper_retains_file_path(tmp_path):
    file_path = tmp_path / "data.jsonl"
    write_json_lines(file_path, [])
    wrapper = StaticJsonWrapper(str(file_path))

    assert wrapper.file_path == str(file_path)


def test_static_json_wrapper_load_is_repeatable(tmp_path):
    dataset = [
        {"tweets": "a", "timestamp": 1, "tweetID": "u", "clusterNo": 2},
    ]
    file_path = tmp_path / "data.jsonl"
    write_json_lines(file_path, dataset)
    wrapper = StaticJsonWrapper(str(file_path))

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


@pytest.mark.parametrize(
    ("records", "expected_texts"),
    [
        (
            [("b", "first b", "2026-01-01"), ("a", "first a", "2026-01-02")],
            ["first b", "first a"],
        ),
        (
            [("a", "later a", "2026-01-03"), ("b", "first b", "2026-01-02"),
             ("a", "first a", "2026-01-01")],
            ["first a", "first b"],
        ),
        (
            [("a", "first a", "2026-01-01"), ("a", "later a", "2026-01-02"),
             ("b", "first b", "2026-01-03"), ("b", "later b", "2026-01-04")],
            ["first a", "first b"],
        ),
        (
            [("a", "first a", "2026-01-01"), ("b", "first b", "2026-01-01"),
             ("a", "later a", "2026-01-01")],
            ["first a", "first b"],
        ),
    ],
    ids=["unique-order", "earliest-occurrence", "all-ids-duplicated", "timestamp-ties"],
)
def test_social_ed_wrapper_retains_first_occurrence(tmp_path, records, expected_texts):
    data = np.full((len(records), 11), None, dtype=object)
    for i, (tweet_id, text, timestamp) in enumerate(records):
        data[i, 0] = tweet_id
        data[i, 1] = text
        data[i, 2] = i
        data[i, 7] = timestamp
    file_path = tmp_path / "social_ed.npy"
    np.save(file_path, data)

    items = list(SocialEDWrapper(str(file_path)).load())
    points = [item for item in items[:-1] if isinstance(item, RawPoint)]

    assert items[-1] is None
    assert len(items) == len(points) + 1
    assert [point.text for point in points] == expected_texts
    assert len({point.text_id for point in points}) == len(points)