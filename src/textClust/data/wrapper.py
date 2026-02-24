from textClust.data.stream import Stream
from textClust.data.point import RawPoint
from typing import List
import json

class StaticJsonWrapper(Stream):
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
    
    def load(self, **kwargs):
        def generator():
            with open(self.file_path, "r") as file:
                for l in file:
                    item = json.loads(l)
                    text = item.get("tweets", None)
                    timestamp = item.get("timestamp", None)
                    id = item.get("tweetID", None)
                    cluster_id = item.get("clusterNo", None)
                    point = RawPoint(text=text, timestamp=timestamp, user=id, cluster_id=cluster_id)
                    yield point
            yield None
        return generator()