"""Point data models used by the textClust pipeline."""

from typing import Optional, List
import numpy as np
import torch

from textClust.online.cluster import Microcluster

class RawPoint:
    """Raw text observation before preprocessing."""

    def __init__(self, text:str, timestamp:Optional[float] = None, text_id: Optional[str] = None,
                 cluster_id: Optional[int] = None, **kwargs):
        """Initialize a raw point.

        Args:
            text: Original input text.
            timestamp: Event time associated with the text.
            text_id: Optional unique identifier for the text.
            cluster_id: Optional ground-truth cluster label.
            **kwargs: Extra metadata stored in `extra`.
        """
        self.text = text
        self.timestamp = timestamp
        self.text_id = text_id
        self.cluster_id = cluster_id
        self.extra = kwargs
        
    def to_microcluster(self) -> Microcluster:
        """Convert the raw point into a single-item microcluster shell."""
        return Microcluster(
            id=None,
            current_time=self.timestamp if self.timestamp is not None else 0,
            n=1,
            embedding={},
            weight=1.0
        )
        
class Point(RawPoint):
    """Preprocessed point with tokens and optional embedding."""

    def __init__(self, text:str, embedding:Optional[dict[str,float]|torch.Tensor] = None, 
                 timestamp:Optional[float] = None, text_id: Optional[str] = None, cluster_id: Optional[int] = None, **kwargs):
        """Initialize a processed point.

        Args:
            text: Original input text.
            processed_text_tokens: Tokenized/normalized terms for TF counting.
            embedding: Optional vector representation of the text.
            timestamp: Event time associated with the text.
            text_id: Optional unique identifier for the text.
            cluster_id: Optional ground-truth cluster label.
            **kwargs: Extra metadata stored in `extra`.
        """
        super().__init__(text, timestamp, text_id, cluster_id, **kwargs)
        self.embedding = embedding
        self.extra = kwargs
        
    def to_microcluster(self) -> Microcluster:
        """Convert the point into a single-item term-frequency microcluster."""
        
        if self.embedding is not None:
            # For simplicity, we won't include the embedding in the microcluster representation here,
            # but it could be incorporated into an EmbeddingMicrocluster if desired.
        
            return Microcluster(
                id=None,
                current_time=self.timestamp if self.timestamp is not None else 0,
                n=1,
                weight=1.0,
                embedding=self.embedding
            )
            
        return Microcluster(
            id=None,
            embedding={},
            current_time=self.timestamp if self.timestamp is not None else 0,
            n=1,
            weight=1.0
        )
        
def point_from_raw_point(raw_point: RawPoint, embedding:Optional[dict[str,float]|torch.Tensor] = None, **kwargs):
    """Create a processed `Point` from a `RawPoint` and derived features."""
    return Point(
        text = raw_point.text,
        timestamp = raw_point.timestamp,
        text_id = raw_point.text_id,
        cluster_id = raw_point.cluster_id,
        embedding = embedding,
        **kwargs
    )