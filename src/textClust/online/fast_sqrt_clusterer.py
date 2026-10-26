from textClust.online.cluster import Microcluster
from textClust.online.clusterer import ThresholdMicroclusterer

import math
import numpy as np
from typing import List, Optional, Sequence, Set, Tuple, final
from copy import deepcopy
import warnings
from sklearn.decomposition import PCA
import graphviz

from textClust.online.fade import ExpFade
from textClust.online.merge import SumMerger
from textClust.online.metrics import CosineDistance

from textClust.online.fast_log_clusterer import FastLogClusterer, FastLogCosineDistance, IDFManager, FastMicrocluster, FastSumMerger

class FastSqrtMicrocluster(FastMicrocluster):        
    def get_norm(self):
        n = self.idf_manager.total_docs if self.idf_manager.total_docs > 0 else 1
        norm_squared = self.weighted_tf_sum * n
        norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
        
        return norm
    
    def get_norm_direct(self):
        norm_squared = np.sum([(v * math.sqrt(self.idf_manager.total_docs/self.idf_manager.term_doc_freq.get(k, 1))) ** 2 for k, v in self.embedding.items()])
        norm = math.sqrt(norm_squared) if norm_squared > 0 else 1.0
        return norm
    
    def add_tf(self, new_tf):
        for term, weight in new_tf.items():
            old_weight = self.embedding.get(term, 0)
            c = self.idf_manager.term_doc_freq.get(term, 1)
            new_weight = old_weight + weight
            
            self.weighted_tf_sum -= old_weight ** 2 * 1/c
            self.weighted_tf_sum += new_weight ** 2 * 1/c
            
            if new_weight > 0:
                self.embedding[term] = new_weight
            else:
                if term in self.embedding:
                    del self.embedding[term]
                    
    def update_norm_by_idf(self, updates:dict):
        relevant_terms = set(self.embedding.keys()) & set(updates.keys())
        
        for term, new_c in {term: updates[term] for term in relevant_terms}.items():
            old_c = self.idf_manager.term_doc_freq.get(term, 1)
            weight = self.embedding.get(term, 0)
            
            self.weighted_tf_sum -= weight ** 2 * 1/old_c
            self.weighted_tf_sum += weight ** 2 * 1/new_c
            
    def calc_sum(self):
        self.weighted_tf_sum = np.sum([v ** 2 * 1/self.idf_manager.term_doc_freq.get(k, 1) for k, v in self.embedding.items()])
        

class FastSqrtCosineDistance(FastLogCosineDistance):
        
    def _inner_product_formula(self, tf_val1: float, tf_val2: float, n: int, term_freq: float) -> float:
        return tf_val1 * tf_val2 * n / term_freq if term_freq > 0 else 0.0

class FastSqrtFader(ExpFade):
    """Fader that also applies fading to FastMicrocluster norms."""

    def __init__(self, decay_rate: float, term_fading: bool = True, use_heuristic: bool = False,
                 min_weight: Optional[float] = None, time_gap: Optional[float] = None):
        super().__init__(decay_rate, term_fading, use_heuristic, min_weight, time_gap)

    def update_mc_internals(self,mc: FastMicrocluster, factor: float):
        assert isinstance(mc, FastSqrtMicrocluster), "FastSqrtFader can only update FastSqrtMicrocluster"
        mc.weighted_tf_sum *= factor

class FastSqrtClusterer(FastLogClusterer):
    def __init__(self, distance_metric: FastSqrtCosineDistance, fader: FastSqrtFader, merger: FastSumMerger, 
                 adaptive_threshold=True, sigma: Optional[float] = 1.0, radius: Optional[float] = None,
                 min_num_mc=5, include_min_for_threshold=False, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, adaptive_threshold=adaptive_threshold, sigma=sigma, radius=radius,
                         min_num_mc=min_num_mc, include_min_for_threshold=include_min_for_threshold, threshold_momentum=threshold_momentum, *args, **kwargs)
        self.microclusters: List[FastSqrtMicrocluster] = [] # type: ignore
        self.idf_manager = IDFManager()
        
    def create_mc(self, mc:Microcluster, current_time: float) -> FastSqrtMicrocluster:
        assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        return FastSqrtMicrocluster(mc.id, mc.embedding, self.idf_manager, current_time, mc.n, mc.weight)
    