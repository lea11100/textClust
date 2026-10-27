from textClust.online.cluster import Microcluster
from textClust.online.clusterer import ThresholdMicroclusterer

import math
import numpy as np
from typing import List, Optional, Sequence, Set, Tuple, final
from copy import deepcopy
import warnings
from sklearn.decomposition import PCA
import graphviz
from collections import defaultdict

from textClust.online.fade import ExpFade
from textClust.online.merge import SumMerger
from textClust.online.metrics import CosineDistance

class FastMicrocluster(Microcluster):
    def __init__(self, id, tf:dict[str, float], idf_manager: "IDFManager", current_time: float, n: int, weight = 1.0):
        super().__init__(id,  current_time, n, embedding=tf, weight=weight)
        self.embedding = tf
        self.idf_manager = idf_manager

class IDFManager:
    def __init__(self):
        self.term_doc_freq = defaultdict(int)  # term -> document frequency
        self.term_log_doc_freq = defaultdict(float)  # term -> log(document frequency)
        self.term_doc_occurrence = defaultdict(set)  # term -> document occurrence
        self.total_docs = 0
        
    def add_new_microcluster(self, mc: FastMicrocluster):
        assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        self.total_docs += 1
        mc.idf_manager = self
        for term in mc.embedding.keys():
            self.term_doc_freq[term] += 1
            self.term_log_doc_freq[term] = math.log(self.term_doc_freq[term])
            self.term_doc_occurrence[term].add(mc.id)
            
    def add_new_merged_microcluster(self, merged_mc: FastMicrocluster, merged_into_mc: FastMicrocluster):
        assert isinstance(merged_mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        assert isinstance(merged_into_mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        
        diff = set(merged_mc.embedding.keys()) - set(merged_into_mc.embedding.keys())
        for term in diff:
            self.term_doc_freq[term] += 1
            self.term_log_doc_freq[term] = math.log(self.term_doc_freq[term])
            self.term_doc_occurrence[term].add(merged_into_mc.id)
        
    def merge_two_clusters(self, merged_mc: FastMicrocluster, merged_into_mc: FastMicrocluster):
        assert isinstance(merged_mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        assert isinstance(merged_into_mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        
        self.total_docs -= 1
        for term in merged_mc.embedding.keys():
            self.term_doc_occurrence[term].remove(merged_mc.id)
            self.term_doc_occurrence[term].add(merged_into_mc.id)
        
        intersection = set(merged_mc.embedding.keys()) & set(merged_into_mc.embedding.keys())
        for term in intersection:
            self.term_doc_freq[term] -= 1
            self.term_log_doc_freq[term] = math.log(self.term_doc_freq[term]) if self.term_doc_freq[term] > 0 else 0.0
    
    def delete_microcluster(self, mc: FastMicrocluster):
        assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        self.total_docs -= 1
        self.delete_terms_from_microcluster(set(mc.embedding.keys()), mc)
    
    def delete_terms_from_microcluster(self, terms_to_delete: Set[str], mc: FastMicrocluster):
        assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        for term in terms_to_delete:
            self.term_doc_freq[term] -= 1
            self.term_log_doc_freq[term] = math.log(self.term_doc_freq[term]) if self.term_doc_freq[term] > 0 else 0.0
            if self.term_doc_freq[term] <= 0:
                del self.term_doc_freq[term]
                del self.term_log_doc_freq[term]
            self.term_doc_occurrence[term].remove(mc.id)
            if not self.term_doc_occurrence[term]:
                del self.term_doc_occurrence[term]


class FastLogCosineDistance(CosineDistance):    
    def _inner_product_formula(self, tf_val1: float, tf_val2: float, n: int, term_freq: float) -> float:
        return tf_val1 * tf_val2 * ((1 + math.log(n / term_freq)) ** 2)
    
    def norm(self, mc: FastMicrocluster) -> float:
        idf_manager = mc.idf_manager
        n = idf_manager.total_docs if idf_manager.total_docs > 0 else 1
        log_n = math.log(n)
        return math.sqrt(sum((tf_val * (1 +log_n - idf_manager.term_log_doc_freq.get(term, 0.0))) ** 2 for term, tf_val in mc.embedding.items()))

    def __call__(self, mc1: FastMicrocluster, mc2: List[FastMicrocluster], *args, **kwargs) -> List[float]:
        assert isinstance(mc1, FastMicrocluster), "FastCosineDistance requires mc1 to be a FastMicrocluster instance"
        for mc in mc2:
            if not isinstance(mc, FastMicrocluster):
                raise ValueError("FastCosineDistance requires a list of FastMicrocluster instances")
        
        idf_manager = mc1.idf_manager
        id_to_position = {mc.id: idx for idx, mc in enumerate(mc2)}
        n = idf_manager.total_docs if idf_manager.total_docs > 0 else 1
        log_n = math.log(n)
        
        dot_products = np.zeros(len(mc2))
        
        for term, tf_val1 in mc1.embedding.items():
            idf = 1 + log_n - idf_manager.term_log_doc_freq.get(term, 0.0)
            for id in idf_manager.term_doc_occurrence.get(term, []):
                idx = id_to_position.get(id)
                if idx is None:
                    continue
                tf_val2 = mc2[idx].embedding.get(term, 0.0)
                dot_products[idx] += tf_val1 * tf_val2 * (idf ** 2)
        
        norm1 = self.norm(mc1)
        norms_other = np.array([self.norm(mc) for mc in mc2])
        denominators = norm1 * norms_other
        similarities = np.divide(dot_products, denominators, out=np.zeros_like(dot_products), where=denominators > 0)
        distances = 1.0 - similarities
        distances = np.clip(distances, 0.0, 1.0)
        return distances.tolist()

class FastSumMerger(SumMerger):
    """Extension of SumMerger that also merges tree structure for TreeMicroclusters."""

    def __call__(self, mc1: FastMicrocluster, mc2: FastMicrocluster, current_time: float) -> FastMicrocluster:
        """Merge two TreeMicroclusters and combine their children."""     
        merged_weight = mc1.weight + mc2.weight
        merged_n = mc1.n + mc2.n
        
        # Important to update the IDFManager before merging embeddings
        if mc2.id == None:
            mc1.idf_manager.add_new_merged_microcluster(merged_mc=mc2, merged_into_mc=mc1)
        else:
            mc1.idf_manager.merge_two_clusters(merged_mc=mc2, merged_into_mc=mc1)
        
        embedding = mc1.embedding.copy()
        for term, tf_val in mc2.embedding.items():
            embedding[term] = embedding.get(term, 0.0) + tf_val        
        merged = FastMicrocluster(mc1.id, embedding, mc1.idf_manager, current_time, merged_n, merged_weight)
        
        return merged


class FastLogClusterer(ThresholdMicroclusterer):
    def __init__(self, distance_metric: FastLogCosineDistance, fader: ExpFade, merger: FastSumMerger, 
                 adaptive_threshold=True, sigma: Optional[float] = 1.0, radius: Optional[float] = None,
                 min_num_mc=5, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, adaptive_threshold=adaptive_threshold, sigma=sigma, radius=radius,
                         min_num_mc=min_num_mc, threshold_momentum=threshold_momentum, *args, **kwargs)
        self.microclusters: List[FastMicrocluster] = [] # type: ignore
        self.idf_manager = IDFManager()

    def _delete_expired(self, current_time: float) -> List[int]:
        """Remove expired clusters and terms, keeping the IDF index in sync."""
        expired_cluster_ids = []
        active_clusters = []
        for mc, is_expired, expired_terms in self._get_expried(current_time):
            assert isinstance(mc, FastMicrocluster)
            if is_expired:
                self.idf_manager.delete_microcluster(mc)
                expired_cluster_ids.append(mc.id)
                continue
            if expired_terms:
                self.idf_manager.delete_terms_from_microcluster(set(expired_terms), mc)
                for term in expired_terms:
                    del mc.embedding[term]
            active_clusters.append(mc)
        self.microclusters[:] = active_clusters
        return expired_cluster_ids

        
    def create_mc(self, mc:Microcluster, current_time: float) -> FastMicrocluster:
        assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
        return FastMicrocluster(mc.id, mc.embedding, self.idf_manager, current_time, mc.n, mc.weight)
    
  
    def update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> int:
        """Update a microcluster."""
        mc=self.create_mc(mc, current_time)
        mc, clusterId = self._update(mc, current_time, *args, **kwargs)
        assert isinstance(mc, FastMicrocluster), "Expected mc to be a FastMicrocluster after update"
        if clusterId is None:
            clusterId = self.pull_new_clusterId()
            mc.id = clusterId
            self.idf_manager.add_new_microcluster(mc)
        return clusterId
    