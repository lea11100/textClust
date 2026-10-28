"""Cluster fading strategies for time-decay behavior."""

import warnings

from textClust.online.cluster import Microcluster
from typing import List, Optional, Tuple

class DampedFade:
    """Base callable for applying time-based fading to a microcluster."""

    def __init__(self, term_fading: bool = False):
        """Initialize fading strategy."""
        self.term_fading = term_fading
    
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool, List[str]]:
        """Apply decay and report `(updated_microcluster, is_expired, expired_terms)`.

        Expired terms remain in the embedding until clusterer cleanup removes them.
        """
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")

class ExpFade(DampedFade):
    """Exponential fading based on elapsed time since last update."""

    def __init__(self, decay_rate, term_fading=False, use_heuristic = False, min_weight: Optional[float] = None, time_gap: Optional[float] = None):
        """Initialize exponential fading parameters."""
        super().__init__(term_fading=term_fading)
        self.decay_rate = decay_rate
        
        if use_heuristic:
            assert time_gap is not None, "time_gap must be provided when use_heuristic is True"
            # Warn if min_weight is provided but will be overridden by heuristic
            if min_weight is not None:
                warnings.warn("Warning: min_weight will be overridden by heuristic calculation based on time_gap.")
            self.min_weight = 2 ** (-1 * decay_rate * time_gap)
        else:
            assert min_weight is not None, "min_weight must be provided when use_heuristic is False"
            self.min_weight = min_weight
        
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool, List[str]]:
        """Exponentially decay cluster weight and optionally term weights."""
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        expired_terms: List[str] = []
        if self.term_fading:
            assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
            for term in mc.embedding:
                mc.embedding[term] *= factor
            expired_terms = [term for term, weight in mc.embedding.items() if weight < self.min_weight]
        
        is_expired = is_expired or (len(mc.embedding) == len(expired_terms))  # Expire if no terms survive
        return mc, is_expired, expired_terms