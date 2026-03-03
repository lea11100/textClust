"""Cluster fading strategies for time-decay behavior."""

from textClust.online.cluster import Microcluster
from typing import Tuple

class DampedFade:
    """Base callable for applying time-based fading to a microcluster."""

    def __init__(self):
        """Initialize fading strategy."""
        pass
    
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool]:
        """Apply decay and return `(updated_microcluster, is_expired)`."""
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")

class ExpFade(DampedFade):
    """Exponential fading based on elapsed time since last update."""

    def __init__(self, decay_rate, term_fading=True, min_weight=0.5):
        """Initialize exponential fading parameters."""
        self.decay_rate = decay_rate
        self.term_fading = term_fading
        self.min_weight = min_weight
        
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool]:
        """Exponentially decay cluster weight and optionally term weights."""
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        if self.term_fading:
            mc.tf = {term: weight * factor for term, weight in mc.tf.items()}
            mc.tf = {term: weight for term, weight in mc.tf.items() if weight > self.min_weight}  # Remove faded terms
        
        is_expired = is_expired or (len(mc.tf) == 0)  # Expire if no terms left
        return mc, is_expired