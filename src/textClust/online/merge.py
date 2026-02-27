from textClust.online.cluster import Microcluster
from textClust.online.fade import DampedFade
from typing import Optional

class Merger:
    def __init__(self):
        pass
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, *args, **kwargs) -> Microcluster:
        raise NotImplementedError("merge() must be implemented by subclass")
    
    
class SumMerger(Merger):
    def __init__(self):
        super().__init__()
        
    def __call__(self, mc1: Microcluster, mc2: Microcluster, current_time: float, fader: Optional[DampedFade] = None) -> Microcluster:
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
        