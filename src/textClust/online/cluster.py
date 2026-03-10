from typing import Optional
import torch

"""Microcluster data structures for online clustering."""

class Microcluster:
    """Base class for microclusters."""

    def __init__(self, id, current_time: float, n: int, weight = 1.0):
        """Initialize a microcluster.

        Args:
            id: Cluster identifier.
            current_time: Last update timestamp.
            n: Number of aggregated points.
            weight: Current cluster weight.
        """
        self.id = id
        self.last_update_time = current_time
        self.weight = weight
        self.n = n
                
    def __eq__(self, value):
        """Return value-based equality for testing and comparisons."""
        if not isinstance(value, Microcluster):
            return False
        return (self.id == value.id and
                self.weight == value.weight and
                self.last_update_time == value.last_update_time and
                self.n == value.n)

class TfIdfMicrocluster(Microcluster):
    """Compact summary of a group of text points."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, weight = 1.0):
        """Initialize a microcluster.

        Args:
            id: Cluster identifier.
            tf: Term-frequency map for represented points.
            current_time: Last update timestamp.
            n: Number of aggregated points.
            weight: Current cluster weight.
        """
        super().__init__(id, current_time, n, weight)
        self.tf = tf
                
    def __eq__(self, value):
        """Return value-based equality for testing and comparisons."""
        return super().__eq__(value) and isinstance(value, TfIdfMicrocluster) and self.tf == value.tf
        
class EmbeddingMicrocluster(Microcluster):
    """Compact summary of a group of text points."""

    def __init__(self, id, current_time: float, n: int, embedding:torch.Tensor, weight = 1.0):
        """Initialize a microcluster.

        Args:
            id: Cluster identifier.
            current_time: Last update timestamp.
            n: Number of aggregated points.
            embedding: Optional embedding summary.
            weight: Current cluster weight.
        """
        super().__init__(id, current_time, n, weight)
        self.embedding = embedding
                
    def __eq__(self, value):
        """Return value-based equality for testing and comparisons."""
        return super().__eq__(value) and isinstance(value, EmbeddingMicrocluster) and torch.equal(self.embedding, value.embedding)