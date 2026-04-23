"""Cluster fading strategies for time-decay behavior."""

import warnings

from textClust.online.cluster import Microcluster, TfIdfMicrocluster, EmbeddingMicrocluster
from typing import Optional, Tuple

class DampedFade:
    """Base callable for applying time-based fading to a microcluster."""

    def __init__(self):
        """Initialize fading strategy."""
        pass
    
    def __call__(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, bool]:
        """Apply decay and return `(updated_microcluster, is_expired)`."""
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")
    
class TfIdfDampedFade(DampedFade):
    def __init__(self):
        """Initialize fading strategy."""
        pass
    
    def __call__(self, mc: TfIdfMicrocluster, current_time: float, *args, **kwargs) -> Tuple[TfIdfMicrocluster, bool]:
        """Apply decay and return `(updated_microcluster, is_expired)`."""
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")
    
class EmbeddingDampedFade(DampedFade):
    def __init__(self):
        """Initialize fading strategy."""
        pass
    
    def __call__(self, mc: EmbeddingMicrocluster, current_time: float, *args, **kwargs) -> Tuple[EmbeddingMicrocluster, bool]:
        """Apply decay and return `(updated_microcluster, is_expired)`."""
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")

class TfIdfExpFade(TfIdfDampedFade):
    """Exponential fading based on elapsed time since last update."""

    def __init__(self, decay_rate, term_fading=True, use_heuristic = False, min_weight: Optional[float] = None, time_gap: Optional[float] = None):
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
        
    def __call__(self, mc: TfIdfMicrocluster, current_time: float, *args, **kwargs) -> Tuple[TfIdfMicrocluster, bool]:
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
    
class EmbeddingExpFader(EmbeddingDampedFade):
    """Exponential fading based on elapsed time since last update."""

    def __init__(self, decay_rate, use_heuristic = False, min_weight: Optional[float] = None, time_gap: Optional[float] = None):
        """Initialize exponential fading parameters."""
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
        
    def __call__(self, mc: EmbeddingMicrocluster, current_time: float, *args, **kwargs) -> Tuple[EmbeddingMicrocluster, bool]:
        """Exponentially decay cluster weight and optionally term weights."""
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        is_expired = is_expired  # Expire if no terms left
        return mc, is_expired