from textClust.online.microcluster import Microcluster
from typing import Tuple

class DampedFade:
    def __init__(self):
        pass
    
    def __call__(self, mc: Microcluster, current_time: float) -> Tuple[Microcluster, bool]:
        raise NotImplementedError("fade_microcluster() must be implemented by subclass")

class ExpFade(DampedFade):
    def __init__(self, decay_rate, term_fading=False, min_weight=0.5):
        self.decay_rate = decay_rate
        self.term_fading = term_fading
        self.min_weight = min_weight
        
    def __call__(self, mc: Microcluster, current_time: float) -> Tuple[Microcluster, bool]:
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        if self.term_fading and not is_expired:
            mc.tf = {term: weight * factor for term, weight in mc.tf.items()}
            mc.tf = {term: weight for term, weight in mc.tf.items() if weight > self.min_weight}  # Remove faded terms
        
        is_expired = is_expired or (len(mc.tf) == 0)  # Expire if no terms left
        return mc, is_expired