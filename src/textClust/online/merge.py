"""Microcluster merge strategies for online updates."""

from textClust.online.cluster import Microcluster
from textClust.online.fade import DampedFade
from typing import Optional

class Merger:
    """Base callable interface for combining two microclusters."""

    def __init__(self):
        """Initialize merger strategy."""
        pass
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        """Merge two microclusters and return the merged result."""
        raise NotImplementedError("merge() must be implemented by subclass")
    
    
class SumMerger(Merger):
    """Merge strategy that sums weights, counts, and term frequencies."""

    def __init__(self):
        """Initialize sum merger."""
        super().__init__()
        
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, fader: Optional[DampedFade] = None) -> Microcluster:
        """Optionally fade inputs then merge into `mc1`'s cluster id."""
        if fader is not None:
            mc1, _ = fader(mc1, current_time)
            mc2, _ = fader(mc2, current_time)
        
        merged_weight = mc1.weight + mc2.weight
        merged_n = mc1.n + mc2.n
        
        merged_tf = mc1.tf
        for term, weight in mc2.tf.items():
            merged_tf[term] = merged_tf.get(term, 0) + weight
        
        return Microcluster(
            id=mc1.id,
            tf=merged_tf,
            current_time=current_time,
            n=merged_n,
            embedding=None,
            weight=merged_weight
        )
        