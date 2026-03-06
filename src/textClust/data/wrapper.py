"""Concrete stream wrappers for loading textClust input data."""

from datetime import datetime

from textClust.data.stream import Stream
from textClust.data.point import RawPoint
from typing import List
import json

class StaticJsonWrapper(Stream):
    """Load newline-delimited JSON records from disk as raw points."""

    def __init__(self, file_path: str, **kwargs):
        """Initialize the JSON wrapper.

        Args:
            file_path: Path to an input JSONL file.
        """
        super().__init__(**kwargs)
        self.file_path = file_path
    
    def load(self, **kwargs):
        """Return a generator of `RawPoint` records from `file_path`."""
        def generator():
            """Yield parsed JSON lines as `RawPoint` objects, then `None`."""
            with open(self.file_path, "r") as file:
                for l in file:
                    item = json.loads(l)
                    text = item.get("tweets", None)
                    timestamp = item.get("timestamp", None)
                    if timestamp is not None:
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        timestamp = timestamp.timestamp()
                    id = item.get("tweetID", None)
                    cluster_id = item.get("clusterNo", None)
                    point = RawPoint(text=text, timestamp=timestamp, text_id=id, cluster_id=cluster_id)
                    yield point
            yield None
        return generator()