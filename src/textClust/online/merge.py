"""Microcluster merge strategies for online updates."""

import torch

from textClust.online.cluster import Microcluster
from textClust.online.fade import DampedFade
from typing import Optional

class Merger:
    """Base callable interface for combining two microclusters."""

    def __init__(self, normalize: bool = False):
        """Initialize merger strategy."""
        self.normalize = normalize
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        """Merge two microclusters and return the merged result."""
        merged = self._merge(mc1, mc2, current_time, *args, **kwargs)
        if self.normalize:
            assert isinstance(merged.embedding, torch.Tensor), "Normalization is only supported for embedding vectors"
            merged.embedding = torch.nn.functional.normalize(merged.embedding, p=2, dim=0)
        return merged

    def _merge(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        """Internal merge logic to be implemented by subclasses."""
        raise NotImplementedError("_merge() must be implemented by subclass")
    
class SumMerger(Merger):
    """Merge strategy that sums weights, counts, and term frequencies."""
        
    def _merge(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        merged_weight = mc1.weight + mc2.weight
        merged_n = mc1.n + mc2.n
        
        merged_embedding = mc1.embedding.copy() if isinstance(mc1.embedding, dict) else mc1.embedding.clone()
        if isinstance(merged_embedding, dict) and isinstance(mc2.embedding, dict):
            for term, weight in mc2.embedding.items():
                merged_embedding[term] = merged_embedding.get(term, 0) + weight
                
        if isinstance(merged_embedding, torch.Tensor) and isinstance(mc2.embedding, torch.Tensor):
            merged_embedding = merged_embedding + mc2.embedding
        
        return Microcluster(
            id=mc1.id,
            current_time=current_time,
            n=merged_n,
            embedding=merged_embedding,
            weight=merged_weight
        )
        
class WeightedSumMerger(Merger):
    """Merge strategy that performs a weighted sum of the two microclusters."""
    def _merge(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        w1 = mc1.weight
        w2 = mc2.weight
        total_weight = w1 + w2
        assert total_weight > 0, "Total weight must be positive for merging"
        
        embedding = {}
        if isinstance(mc1.embedding, dict) and isinstance(mc2.embedding, dict):
            embedding = {term: (w1 * mc1.embedding.get(term, 0) + w2 * mc2.embedding.get(term, 0)) / total_weight
                         for term in set(mc1.embedding.keys()).union(mc2.embedding.keys())}
        
        if isinstance(mc1.embedding, torch.Tensor) and isinstance(mc2.embedding, torch.Tensor):
            embedding = (w1 * mc1.embedding + w2 * mc2.embedding) / total_weight
      
        total_n = mc1.n + mc2.n
            
        merged = Microcluster(mc1.id, current_time, n=total_n, embedding=embedding, weight=total_weight)
        return merged
        