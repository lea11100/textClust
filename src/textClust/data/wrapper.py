from textClust.data.stream import Stream
from textClust.data.point import RawPoint
from typing import List

class StaticJsonWrapper(Stream):
    def __init__(self, dataset: List, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
    
    def load(self, **kwargs):
        def generator():
            for item in self.dataset:
                text = item.get("tweets", None)
                timestamp = item.get("timestamp", None)
                id = item.get("tweetID", None)
                cluster_id = item.get("clusterNo", None)
                point = RawPoint(text=text, timestamp=timestamp, user=id, cluster_id=cluster_id)
                yield point
            yield None
        return generator()