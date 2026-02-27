from typing import Any
import numpy as np

from textClust.online.cluster import Microcluster

class DistanceMetric:
    def __init__(self):
        self.requires_idf = False
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, *args: Any, **kwds: Any) -> float:
        raise NotImplementedError("DistanceMetric must implement __call__()")
    
class TfidfCosineDistance(DistanceMetric):
    def __init__(self):
        super().__init__()
        self.requires_idf = True
        
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, idf: dict[str, float], **kwds: Any) -> float:
        # Compute cosine distance between two microclusters based on their TF-IDF vectors
        tf1 = mc1.tf
        tf2 = mc2.tf
       
        # Create uniform embeddings
        intersection = set(tf1.keys()) & set(tf2.keys())
        
        if not intersection:
            return 1.0
        
        inner_product = 0.0
        
        for term in intersection:
            tf_val1 = tf1[term]
            tf_val2 = tf2[term]
            
            if idf and term in idf:
                inner_product += idf[term] ** 2 * tf_val1 * tf_val2
            else:
                inner_product += tf_val1 * tf_val2
                
        tfidf1, tfidf2 = self._make_tfidf_vectors(tf1, tf2, idf)
        norm1 = np.linalg.norm(tfidf1).astype(float)
        norm2 = np.linalg.norm(tfidf2).astype(float)

        similarity = inner_product / (norm1 * norm2)
        distance = 1 - similarity
        
        return distance
    
    def _make_tfidf_vectors(self, tf1: dict[str, float], tf2: dict[str, float], idf: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        if idf:
            tfidf1 = [(idf.get(k, 1.0) * v) for k, v in tf1.items()]
            tfidf2 = [(idf.get(k, 1.0) * v) for k, v in tf2.items()]
        else:
            tfidf1 = [v for v in tf1.values()]
            tfidf2 = [v for v in tf2.values()]
            
        return np.array(tfidf1), np.array(tfidf2)