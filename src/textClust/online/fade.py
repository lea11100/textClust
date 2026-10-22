"""Cluster fading strategies for time-decay behavior."""

import warnings

from textClust.online.cluster import Microcluster
from typing import Optional, Tuple

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

    def __init__(self, decay_rate, term_fading=False, use_heuristic = False, min_weight: Optional[float] = None, time_gap: Optional[float] = None):
        """Initialize exponential fading parameters."""
        self.decay_rate = decay_rate
        self.term_fading = term_fading
        
        if use_heuristic:
            assert time_gap is not None, "time_gap must be provided when use_heuristic is True"
            # Warn if min_weight is provided but will be overridden by heuristic
            if min_weight is not None:
                warnings.warn("Warning: min_weight will be overridden by heuristic calculation based on time_gap.")
            self.min_weight = 2 ** (-1 * decay_rate * time_gap)
        else:
            assert min_weight is not None, "min_weight must be provided when use_heuristic is False"
            self.min_weight = min_weight
        
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool]:
        """Exponentially decay cluster weight and optionally term weights."""
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        if self.term_fading:
            assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
            mc.embedding = {term: weight * factor for term, weight in mc.embedding.items()}
            mc.embedding = {term: weight for term, weight in mc.embedding.items() if weight > self.min_weight}  # Remove faded terms
        
        is_expired = is_expired or (len(mc.embedding) == 0)  # Expire if no terms left
        return mc, is_expired