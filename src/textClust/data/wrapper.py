"""Concrete stream wrappers for loading textClust input data."""

from datetime import datetime

from textClust.data.stream import Stream
from textClust.data.point import RawPoint
from typing import List, Optional
import json

class StaticJsonWrapper(Stream):
    """Load newline-delimited JSON records from disk as raw points."""

    def __init__(self, file_path: str, max_samples: Optional[int] = None, do_check: bool = False, **kwargs):
        """Initialize the JSON wrapper.

        Args:
            file_path: Path to an input JSONL file.
            max_samples: Maximum number of samples to load. If `None`, load all samples.
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples
        self.file_path = file_path
        self.do_check = do_check
    
    def load(self, **kwargs):
        """Return a generator of `RawPoint` records from `file_path`."""
        def generator():
            """Yield parsed JSON lines as `RawPoint` objects, then `None`."""
            i = 0
            with open(self.file_path, "r") as file:
                for l in file:
                    if self.max_samples is not None and i >= self.max_samples:
                        break
                    i += 1
                    item = json.loads(l)
                    text = item.get("tweets", None)
                    timestamp = item.get("timestamp", None)
                    if timestamp is not None:
                        if isinstance(timestamp, str):
                            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            timestamp = timestamp.timestamp()
                    id = item.get("tweetID", None)
                    cluster_id = item.get("clusterNo", None)
                    point = RawPoint(text=text, timestamp=timestamp, text_id=id, cluster_id=cluster_id)
                    yield point
            yield None
            
        if self.do_check:
            self._check()
        return generator()
    
    def _check(self):
        """Check if tweetIDs are unique"""
        tweet_ids = set()
        with open(self.file_path, "r") as file:
            for l in file:
                item = json.loads(l)
                text_id = item.get("tweetID", None)
                if text_id is not None:
                    if text_id in tweet_ids:
                        raise ValueError(f"Duplicate tweetID found: {text_id}")
                    tweet_ids.add(text_id)