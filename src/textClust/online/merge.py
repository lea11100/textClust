"""Microcluster merge strategies for online updates."""

from textClust.online.cluster import Microcluster, TfIdfMicrocluster, EmbeddingMicrocluster
from textClust.online.fade import TfIdfDampedFade
from typing import Optional

class Merger:
    """Base callable interface for combining two microclusters."""

    def __init__(self):
        """Initialize merger strategy."""
        pass
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        """Merge two microclusters and return the merged result."""
        raise NotImplementedError("merge() must be implemented by subclass")
    
class TfIdfMerger(Merger):
    """Base callable interface for combining two microclusters."""

    def __init__(self):
        """Initialize merger strategy."""
        pass
    
    def __call__(self, mc1: TfIdfMicrocluster, mc2: TfIdfMicrocluster, current_time: float, *args, **kwargs) -> TfIdfMicrocluster:
        """Merge two microclusters and return the merged result."""
        raise NotImplementedError("merge() must be implemented by subclass")
    
class EmbeddingMerger(Merger):
    """Base callable interface for combining two microclusters."""

    def __init__(self):
        """Initialize merger strategy."""
        pass
    
    def __call__(self, mc1: EmbeddingMicrocluster, mc2: EmbeddingMicrocluster, current_time: float, *args, **kwargs) -> EmbeddingMicrocluster:
        """Merge two microclusters and return the merged result."""
        raise NotImplementedError("merge() must be implemented by subclass")
    
    
class SumMerger(TfIdfMerger):
    """Merge strategy that sums weights, counts, and term frequencies."""

    def __init__(self):
        """Initialize sum merger."""
        super().__init__()
        
    def __call__(self, mc1: TfIdfMicrocluster, mc2: TfIdfMicrocluster, current_time: float, fader: Optional[TfIdfDampedFade] = None, *args, **kwargs) -> TfIdfMicrocluster:
        """Optionally fade inputs then merge into `mc1`'s cluster id."""
        if fader is not None:
            mc1, _ = fader(mc1, current_time)
            mc2, _ = fader(mc2, current_time)
        
        merged_weight = mc1.weight + mc2.weight
        merged_n = mc1.n + mc2.n
        
        merged_tf = mc1.tf
        for term, weight in mc2.tf.items():
            merged_tf[term] = merged_tf.get(term, 0) + weight
        
        return TfIdfMicrocluster(
            id=mc1.id,
            tf=merged_tf,
            current_time=current_time,
            n=merged_n,
            weight=merged_weight
        )
        
class WeightedSumMerger(EmbeddingMerger):

    def __call__(self, mc1: EmbeddingMicrocluster, mc2: EmbeddingMicrocluster, current_time: float) -> EmbeddingMicrocluster:
        w1 = mc1.weight
        w2 = mc2.weight
        total_weight = w1 + w2
        assert total_weight > 0, "Total weight must be positive for merging"
        embedding = (w1 * mc1.embedding + w2 * mc2.embedding) / total_weight
        total_n = mc1.n + mc2.n
            
        merged = EmbeddingMicrocluster(mc1.id, mc1.last_update_time, total_n, embedding, total_weight)
        return merged
        