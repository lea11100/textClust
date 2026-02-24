from typing import Optional, List
import numpy as np

class RawPoint:
    def __init__(self, text:str, timestamp:Optional[int] = None, user: Optional[str] = None,
                 cluster_id: Optional[int] = None, **kwargs):
        self.text = text
        self.timestamp = timestamp
        self.user = user
        self.cluster_id = cluster_id
        self.extra = kwargs
        
class Point(RawPoint):
    def __init__(self, text:str, processed_text_tokens:Optional[List[str]] = None, embedding:Optional[np.ndarray] = None, 
                 timestamp:Optional[int] = None, user: Optional[str] = None, cluster_id: Optional[int] = None, **kwargs):
        super().__init__(text, timestamp, user, cluster_id, **kwargs)
        self.processed_text_tokens = processed_text_tokens
        self.embedding = embedding
        self.extra = kwargs
        
def point_from_raw_point(raw_point: RawPoint, processed_text_tokens:Optional[List[str]] = None, embedding:Optional[np.ndarray] = None, **kwargs):
    return Point(
        text = raw_point.text,
        processed_text_tokens = processed_text_tokens,
        timestamp = raw_point.timestamp,
        user = raw_point.user,
        cluster_id = raw_point.cluster_id,
        embedding = embedding,
        **kwargs
    )