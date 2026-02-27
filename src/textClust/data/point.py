from typing import Optional, List
import numpy as np

from textClust.online.cluster import Microcluster

class RawPoint:
    def __init__(self, text:str, timestamp:Optional[int] = None, text_id: Optional[str] = None,
                 cluster_id: Optional[int] = None, **kwargs):
        self.text = text
        self.timestamp = timestamp
        self.text_id = text_id
        self.cluster_id = cluster_id
        self.extra = kwargs
        
    def to_microcluster(self) -> Microcluster:
        return Microcluster(
            id=None,
            tf={},  # Term frequencies can be computed later based on processed tokens
            current_time=self.timestamp if self.timestamp is not None else 0,
            n=1,
            weight=1.0
        )
        
class Point(RawPoint):
    def __init__(self, text:str, processed_text_tokens:Optional[List[str]] = None, embedding:Optional[np.ndarray] = None, 
                 timestamp:Optional[int] = None, text_id: Optional[str] = None, cluster_id: Optional[int] = None, **kwargs):
        super().__init__(text, timestamp, text_id, cluster_id, **kwargs)
        self.processed_text_tokens = processed_text_tokens
        self.embedding = embedding
        self.extra = kwargs
        
    def to_microcluster(self) -> Microcluster:
        tf = {}
        if self.processed_text_tokens is not None:
            for token in self.processed_text_tokens:
                tf[token] = tf.get(token, 0) + 1
        
        return Microcluster(
            id=None,
            tf=tf,
            current_time=self.timestamp if self.timestamp is not None else 0,
            n=1,
            embedding=self.embedding,
            weight=1.0
        )
        
def point_from_raw_point(raw_point: RawPoint, processed_text_tokens:Optional[List[str]] = None, embedding:Optional[np.ndarray] = None, **kwargs):
    return Point(
        text = raw_point.text,
        processed_text_tokens = processed_text_tokens,
        timestamp = raw_point.timestamp,
        text_id = raw_point.text_id,
        cluster_id = raw_point.cluster_id,
        embedding = embedding,
        **kwargs
    )