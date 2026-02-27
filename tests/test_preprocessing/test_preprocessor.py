import pytest

from textClust.data.point import RawPoint
from textClust.preprocessing.preprocessor import TFIDFPreprocessor


def test_preprocess_text_default_options(monkeypatch):
    monkeypatch.setattr("textClust.preprocessing.preprocessor.stopwords.words", lambda _: ["and", "the"])
    preprocessor = TFIDFPreprocessor()

    text = "Check out https://example.com #Topic @user the and a 42 ok!"
    tokens = preprocessor.preprocess_text(text)

    assert tokens == ["check", "out", "topic", "ok"]


def test_preprocess_text_can_remove_hashtags(monkeypatch):
    monkeypatch.setattr("textClust.preprocessing.preprocessor.stopwords.words", lambda _: [])
    preprocessor = TFIDFPreprocessor(hashtag=False)

    tokens = preprocessor.preprocess_text("#Tag")

    assert tokens == []


def test_preprocess_text_excludes_tokens_case_insensitive(monkeypatch):
    monkeypatch.setattr("textClust.preprocessing.preprocessor.stopwords.words", lambda _: [])
    preprocessor = TFIDFPreprocessor(exclude_tokens=["Topic", "OK"])

    tokens = preprocessor.preprocess_text("topic ok other")

    assert tokens == ["other"]


def test_preprocess_text_with_stemming(monkeypatch):
    monkeypatch.setattr("textClust.preprocessing.preprocessor.stopwords.words", lambda _: [])
    preprocessor = TFIDFPreprocessor(stemming=True, stopword_removal=False)

    tokens = preprocessor.preprocess_text("running runs runner")

    assert tokens == ["run", "run", "runner"]


def test_call_returns_point_with_processed_tokens(monkeypatch):
    monkeypatch.setattr("textClust.preprocessing.preprocessor.stopwords.words", lambda _: [])
    preprocessor = TFIDFPreprocessor(stopword_removal=False)
    raw_point = RawPoint(text="Hello world")

    point = preprocessor(raw_point)

    assert point.text == "Hello world"
    assert point.processed_text_tokens == ["hello", "world"]
