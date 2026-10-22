from typing import Optional
import torch

"""Microcluster data structures for online clustering."""

class Microcluster:
    """Base class for microclusters."""

    def __init__(self, id, current_time: float, n: int, embedding: dict[str, float]|torch.Tensor, weight = 1.0):
        """Initialize a microcluster.

        Args:
            id: Cluster identifier.
            current_time: Last update timestamp.
            n: Number of aggregated points.
            embedding: Embedding as a TF dictionary (str -> float) or tensor.
            weight: Current cluster weight.
        """
        self.id = id
        self.last_update_time = current_time
        self.weight = weight
        self.n = n
        self.embedding = embedding
                
    def __eq__(self, value):
        """Return value-based equality for testing and comparisons."""
        if not isinstance(value, Microcluster):
            return False
        base_equality = (self.id == value.id and
                self.weight == value.weight and
                self.last_update_time == value.last_update_time and
                self.n == value.n)
        if not base_equality:
            return False
        if isinstance(self.embedding, dict) and isinstance(value.embedding, dict):
            return self.embedding == value.embedding
        if isinstance(self.embedding, torch.Tensor) and isinstance(value.embedding, torch.Tensor):
            return torch.equal(self.embedding, value.embedding)
        return False