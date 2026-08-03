"""Concrete stream wrappers for loading textClust input data."""

from datetime import datetime
import numpy as np

from textClust.data.stream import Stream
from textClust.data.point import RawPoint
from typing import List, Optional
import json

class SocialEDWrapper(Stream):
    """Load a static numpy array from disk as raw points."""

    def __init__(self, file_path: str, max_samples: Optional[int] = None, random_start_seed: Optional[int] = None, **kwargs):
        """Initialize the numpy wrapper.

        Args:
            file_path: Path to an input numpy file.
            max_samples: Maximum number of samples to load. If `None`, load all samples.
            random_start_seed: Seed for random number generation.
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples
        self.file_path = file_path
        self.random_start_seed = random_start_seed
    
    def load(self, **kwargs):
        """Return a generator of `RawPoint` records from `file_path`."""
        # Columns: tweet_id, text, event_id, words, filtered_words, entities, user_id, created_at, urls, hashtags, user_mentions
        data = np.load(self.file_path, allow_pickle=True)
        start = 0
        end = data.shape[0]
        if self.random_start_seed is not None:
            np.random.seed(self.random_start_seed)
            start = np.random.randint(0, data.shape[0])
        if self.max_samples is not None:
            end = min(start + self.max_samples, data.shape[0])
        data = data[start:end]
        for i in range(data.shape[0]):
            yield RawPoint(text=data[i, 1], timestamp=None, text_id=data[i, 0], cluster_id=data[i, 2])
        yield None

class StaticJsonWrapper(Stream):
    """Load newline-delimited JSON records from disk as raw points."""

    def __init__(self, file_path: str, max_samples: Optional[int] = None, random_start_seed: Optional[int] = None, do_check: bool = False, **kwargs):
        """Initialize the JSON wrapper.

        Args:
            file_path: Path to an input JSONL file.
            max_samples: Maximum number of samples to load. If `None`, load all samples.
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples
        self.file_path = file_path
        self.random_start_seed = random_start_seed
        self.do_check = do_check
    
    def load(self, **kwargs):
        """Return a generator of `RawPoint` records from `file_path`."""
        def generator():
            """Yield parsed JSON lines as `RawPoint` objects, then `None`."""
            i = 0
            file_length = sum(1 for _ in open(self.file_path))
            start = 0
            end = None
            if self.random_start_seed is not None:
                np.random.seed(self.random_start_seed)
                start = np.random.randint(0, file_length)
            if self.max_samples is not None:
                end = min(start + self.max_samples, file_length)
            with open(self.file_path, "r") as file:
                for l in file:
                    if i < start:
                        i += 1
                        continue
                    if end is not None and i >= end:
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
                    cluster_id = item.get("clusterNo", None) or item.get("topic", None)
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