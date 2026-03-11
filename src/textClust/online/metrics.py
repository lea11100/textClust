"""Distance metrics used for microcluster similarity computations."""

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from textClust.online.cluster import EmbeddingMicrocluster, TfIdfMicrocluster, Microcluster

class DistanceMetric:
    """Base callable interface for microcluster distance metrics."""

    def __init__(self):
        """Initialize metric capabilities."""
        self.requires_idf = False
        self.support_batch = False
    
    def __call__(self, mc1: Microcluster, mc2: Microcluster, *args: Any, **kwds: Any) -> float:
        """Return a non-negative distance between two microclusters."""
        raise NotImplementedError("DistanceMetric must implement __call__()")
    
class TfIdfDistanceMetric(DistanceMetric):
    """Base class for distance metrics over TF-IDF microclusters."""

    def __init__(self):
        """Initialize metric capabilities."""
        super().__init__()
        self.requires_idf = False
    
    def __call__(self, mc1: TfIdfMicrocluster, mc2: TfIdfMicrocluster, *args: Any, **kwds: Any) -> float:
        """Return a non-negative distance between two microclusters."""
        raise NotImplementedError("DistanceMetric must implement __call__()")
    
class EmbeddingDistanceMetric(DistanceMetric):
    """Base class for distance metrics over embedding microclusters."""

    def __init__(self):
        """Initialize metric capabilities."""
        super().__init__()
        self.requires_idf = False
    
    def __call__(self, mc1: EmbeddingMicrocluster, mc2: EmbeddingMicrocluster, *args: Any, **kwds: Any) -> float:
        """Return a non-negative distance between two microclusters."""
        raise NotImplementedError("DistanceMetric must implement __call__()")
    
class TfidfCosineDistance(TfIdfDistanceMetric):
    """Cosine distance over TF-IDF-weighted term-frequency vectors."""

    def __init__(self):
        """Initialize cosine metric and declare IDF dependency."""
        super().__init__()
        self.requires_idf = True
        
    
    def __call__(self, mc1: TfIdfMicrocluster, mc2: TfIdfMicrocluster, idf: dict[str, float], **kwds: Any) -> float:
        """Compute TF-IDF cosine distance in the interval `[0, 1]`."""
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
        """Build weighted vectors from term-frequency mappings."""
        if idf:
            tfidf1 = [(idf.get(k, 1.0) * v) for k, v in tf1.items()]
            tfidf2 = [(idf.get(k, 1.0) * v) for k, v in tf2.items()]
        else:
            tfidf1 = [v for v in tf1.values()]
            tfidf2 = [v for v in tf2.values()]
            
        return np.array(tfidf1), np.array(tfidf2)
    
class EmbeddingCosineDistance(EmbeddingDistanceMetric):
    def __init__(self):
        super().__init__()
        self.requires_idf = False
        self.support_batch = True
        
    def __call__(self, mc1: EmbeddingMicrocluster, mc2: EmbeddingMicrocluster, *args, **kwargs) -> float:
        embedding1 = mc1.embedding
        embedding2 = mc2.embedding
        
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
        distance = 1 - similarity
        
        return distance
    
    def batch_distance(self, mc1: EmbeddingMicrocluster, other_mcs: list[EmbeddingMicrocluster], *args, **kwargs) -> list[float]:
        embedding1 = mc1.embedding
        if isinstance(other_mcs, list):
            distances = []
            emb = torch.stack([mc.embedding for mc in other_mcs])
            similarities = F.cosine_similarity(embedding1.unsqueeze(0), emb, dim=1)
            distances = (1 - similarities).tolist()
            return distances
        else:
            embedding2 = mc2.embedding        
            similarity = F.cosine_similarity(embedding1, embedding2, dim=0).item()
            distance = 1 - similarity
        
        return distance