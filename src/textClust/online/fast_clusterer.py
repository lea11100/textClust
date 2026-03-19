from textClust.online.cluster import Microcluster, TfIdfMicrocluster
from textClust.online.clusterer import Microclusterer, TfIdfAdaptiveThresholdMicroclusterer

import math
import numpy as np
from typing import List, Optional, Sequence, Set, Tuple, final
from copy import deepcopy
import warnings
from sklearn.decomposition import PCA
import graphviz

from textClust.online.fade import TfIdfExpFade, TfIdfDampedFade
from textClust.online.merge import TfIdfMerger, SumMerger
from textClust.online.metrics import DistanceMetric, TfidfCosineDistance

class IDFManager:
    def __init__(self):
        self.term_doc_freq = {}
        self.total_docs = 0
        self.pending_updates = {}
        
    def add_pending_updates(self, updates:dict):
        for term in updates.keys():
            self.pending_updates[term] = updates[term]
            
    def update_idf(self):
        for term, new_c in self.pending_updates.items():
            self.term_doc_freq[term] = new_c
        self.pending_updates = {}
        
    def recalculate(self, microclusters: List["FastMicrocluster"]):
        self.term_doc_freq = {}
        for mc in microclusters:
            for term in mc.tf.keys():
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1
                
        self.total_docs = len(microclusters)

class FastMicrocluster(TfIdfMicrocluster):
    def __init__(self, id, tf:dict[str, float], idf_manager: IDFManager, current_time: float, n: int, weight = 1.0):
        super().__init__(id, tf, current_time, n, weight)
       
        self.idf_manager = idf_manager
        self.calc_sum()
        
    def get_norm(self):
        sqrt_n = math.sqrt(self.idf_manager.total_docs) if self.idf_manager.total_docs > 0 else 1
        norm_squared = self.weighted_tf_sum * sqrt_n
        norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
        
        return norm
    
    def get_norm_direct(self):
        norm_squared = np.sum([(v * math.sqrt(math.sqrt(self.idf_manager.total_docs/self.idf_manager.term_doc_freq.get(k, 1)))) ** 2 for k, v in self.tf.items()])
        norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
        return norm
    
    def add_tf(self, new_tf):
        for term, weight in new_tf.items():
            old_weight = self.tf.get(term, 0)
            c = self.idf_manager.term_doc_freq.get(term, 1)
            new_weight = old_weight + weight
            
            self.weighted_tf_sum -= old_weight ** 2 * 1/math.sqrt(c)
            self.weighted_tf_sum += new_weight ** 2 * 1/math.sqrt(c)
            
            if new_weight > 0:
                self.tf[term] = new_weight
            else:
                if term in self.tf:
                    del self.tf[term]
                    
    def update_norm_by_idf(self, updates:dict):
        relevant_terms = set(self.tf.keys()) & set(updates.keys())
        
        for term, new_c in {term: updates[term] for term in relevant_terms}.items():
            old_c = self.idf_manager.term_doc_freq.get(term, 1)
            weight = self.tf.get(term, 0)
            
            self.weighted_tf_sum -= weight ** 2 * 1/math.sqrt(old_c)
            self.weighted_tf_sum += weight ** 2 * 1/math.sqrt(new_c)
            
    def calc_sum(self):
        self.weighted_tf_sum = np.sum([v ** 2 * 1/math.sqrt(self.idf_manager.term_doc_freq.get(k, 1)) for k, v in self.tf.items()])
        

class FastCosineDistance(TfidfCosineDistance):
    def __init__(self):
        super().__init__()
        self.requires_idf = False
        
    def __call__(self, mc1: FastMicrocluster, mc2: FastMicrocluster, *args, **kwargs) -> float:
        if not isinstance(mc1, FastMicrocluster) or not isinstance(mc2, FastMicrocluster):
            raise ValueError("FastCosineDistance requires FastMicrocluster instances")
        
        tf1 = mc1.tf
        tf2 = mc2.tf
              
        if not tf1 or not tf2:
            return 1.0
        
        # Create uniform embeddings
        intersection = set(tf1.keys()) & set(tf2.keys())
        
        if not intersection:
            return 1.0
        
        inner_product = 0.0
        sqrt_n = math.sqrt(mc1.idf_manager.total_docs) if mc1.idf_manager.total_docs > 0 else 1
        
        for term in intersection:
            tf_val1 = tf1[term]
            tf_val2 = tf2[term]
            
            if term in mc1.idf_manager.term_doc_freq:
                inner_product += tf_val1 * tf_val2 * sqrt_n / math.sqrt(mc1.idf_manager.term_doc_freq[term])
                
        norm1 = mc1.get_norm()
        norm2 = mc2.get_norm()

        similarity = inner_product / (norm1 * norm2)
        
        distance = min(1, 1 - similarity)
        distance = max(0, distance)
        return distance

class FastFader(TfIdfExpFade):
    """Fader that also applies fading to FastMicrocluster norms."""
    
    def __init__(self, decay_rate: float, term_fading: bool = True, min_weight: float = 0.5):
        super().__init__(decay_rate, term_fading, min_weight)

    def __call__(self, mc: FastMicrocluster, current_time: float) -> tuple[FastMicrocluster, bool]:
        factor = 2 ** (- self.decay_rate * (current_time - mc.last_update_time))
        mc.weight *= factor
        mc.last_update_time = current_time
        is_expired = mc.weight < self.min_weight
        
        if self.term_fading:
            mc.weighted_tf_sum *= factor
            mc.add_tf({term: -weight for term, weight in mc.tf.items() if weight < self.min_weight})
            mc.idf_manager.add_pending_updates({term: mc.idf_manager.term_doc_freq.get(term, 1) - 1 for term, weight in mc.tf.items() if weight < self.min_weight})
        
        is_expired = is_expired or (len(mc.tf) == 0)  # Expire if no terms left
        return mc, is_expired

class FastSumMerger(SumMerger):
    """Extension of SumMerger that also merges tree structure for TreeMicroclusters."""

    def __call__(self, mc1: FastMicrocluster, mc2: FastMicrocluster, current_time: float, fader: Optional[FastFader] = None) -> TfIdfMicrocluster:
        """Merge two TreeMicroclusters and combine their children."""     
        if fader is not None:
            mc1, _ = fader(mc1, current_time)
            mc2, _ = fader(mc2, current_time)
            assert isinstance(mc1, FastMicrocluster) and isinstance(mc2, FastMicrocluster), "Fader must return FastMicroclusters"
        
        merged_weight = mc1.weight + mc2.weight
        merged_n = mc1.n + mc2.n
        common_terms = set(mc1.tf.keys()) & set(mc2.tf.keys())
        new_terms = set(mc2.tf.keys()) - set(mc1.idf_manager.term_doc_freq.keys())
        mc1.add_tf(mc2.tf)
        
        mc1.idf_manager.add_pending_updates({term: max(mc1.idf_manager.term_doc_freq.get(term, 1) - 1,1) for term in common_terms})
        mc1.idf_manager.add_pending_updates({term: 1 for term in new_terms})
        
        merged = FastMicrocluster(mc1.id, mc1.tf, mc1.idf_manager, mc1.last_update_time, merged_n, merged_weight)
        
        return merged

class FastClusterer(TfIdfAdaptiveThresholdMicroclusterer):
    def __init__(self, distance_metric: FastCosineDistance, fader: FastFader, merger: FastSumMerger, sigma=1.0,
                 min_num_mc=5, include_min_for_threshold=False, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, sigma=sigma, min_num_mc=min_num_mc,
                         include_min_for_threshold=include_min_for_threshold, threshold_momentum=threshold_momentum, *args, **kwargs)
        self.microclusters: List[FastMicrocluster] = [] # type: ignore
        self.idf_manager = IDFManager()
        
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[FastMicrocluster, Optional[int]]:
        assert isinstance(mc, TfIdfMicrocluster), "FastClusterer can only update TfIdfMicrocluster"
        mc = FastMicrocluster(mc.id, mc.tf, self.idf_manager, mc.last_update_time, mc.n, mc.weight)
        new_mc, id = super()._update(mc, current_time, *args, **kwargs)
        assert isinstance(new_mc, FastMicrocluster), "Updated microcluster must be a FastMicrocluster"
        if id is None:
            self.idf_manager.total_docs += 1
            self.idf_manager.add_pending_updates({term: self.idf_manager.term_doc_freq.get(term, 0) + 1 for term in new_mc.tf.keys()})
        if len(self.idf_manager.pending_updates) > 0:
            for mc in self.microclusters:
                mc.update_norm_by_idf(self.idf_manager.pending_updates)
            self.idf_manager.update_idf()
        return new_mc, id
            
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List, dict]:
        expired_ids, merge_history = super().clean_up(current_time, *args, **kwargs)
        self.idf_manager.recalculate(self.microclusters)
        for mc in self.microclusters:
            mc.calc_sum()
        return expired_ids, merge_history
    
    def update_with_merge_history(self, merge_history: Optional[dict[int, int]], current_time: float, *args, **kwargs):
        super().update_with_merge_history(merge_history, current_time, *args, **kwargs)
        self.idf_manager.recalculate(self.microclusters)
        for mc in self.microclusters:
            mc.calc_sum()
    