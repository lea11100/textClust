"""Distance metrics used for microcluster similarity computations."""

from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import math

from textClust.online.cluster import Microcluster

class DistanceMetric:
    """Base callable interface for microcluster distance metrics."""

    def __init__(self):
        """Initialize metric capabilities."""
    
    def __call__(self, mc1: Microcluster, mc2: List[Microcluster], *args: Any, **kwds: Any) -> List[float]:
        """Return a non-negative distance between two microclusters."""
        raise NotImplementedError("DistanceMetric must implement __call__()")
    
class CosineDistance(DistanceMetric):
    """Cosine distance."""

    def __init__(self, assume_normalized: bool = False):
        """Initialize cosine metric and declare IDF dependency."""
        super().__init__()
        self.assume_normalized = assume_normalized
        
    
    def __call__(self, mc1: Microcluster, mc2: List[Microcluster], idf: Optional[dict[str, float]]=None, n:Optional[int]=None, **kwds: Any) -> List[float]:
        """Compute TF-IDF cosine distance in the interval `[0, 1]`."""
        # Compute cosine distance between two microclusters based on their TF-IDF vectors
        if idf is not None and n is not None and isinstance(mc1.embedding, dict):
            return self.compute_1_n_idf(mc1, mc2, idf, n, **kwds)
        else:
            return self.compute_1_n(mc1, mc2, **kwds)
    
    def compute_1_n_idf(self, mc1: Microcluster, mc2_list: List[Microcluster], idf: dict[str, float], n:int, **kwds: Any) -> List[float]:
        assert isinstance(mc1.embedding, dict), "Expected mc1 to be a TfIdfMicrocluster"
        assert self.assume_normalized is False, "IDF-based cosine distance does not support assume_normalized=True"
        tfidf1 = {term: weight * idf.get(term, 1 + math.log(n)) for term, weight in mc1.embedding.items()}
        norm1 = np.linalg.norm(list(tfidf1.values())).astype(float)            
        
        distances = []
        for mc2 in mc2_list:
            assert isinstance(mc2.embedding, dict), "Expected mc2 to be a TfIdfMicrocluster"
            tfidf2 = {term: weight * idf.get(term, 1 + math.log(n)) for term, weight in mc2.embedding.items()}
            
            intersection = set(tfidf1.keys()) & set(tfidf2.keys())
            if not intersection:
                distances.append(1.0)
                continue
            
            inner_product = sum(tfidf1[term] * tfidf2[term] for term in intersection)
            norm2 = np.linalg.norm(list(tfidf2.values())).astype(float)
            
            similarity = inner_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            distance = 1 - similarity
            distances.append(distance)
        return distances
    
    
    def compute_1_n(self, mc1: Microcluster, mc2_list: list[Microcluster], **kwds: Any) ->  list[float]:
        if isinstance(mc1.embedding, torch.Tensor):
            m2_embeddings = torch.stack([mc2.embedding for mc2 in mc2_list if isinstance(mc2.embedding, torch.Tensor)], dim=0)
            if self.assume_normalized:
                similarities = torch.inner(mc1.embedding.unsqueeze(0), m2_embeddings).reshape(-1)
            else:
                similarities = F.cosine_similarity(mc1.embedding.unsqueeze(0), m2_embeddings, dim=1)
            distances = (1 - similarities).tolist()
            return distances
        
        return [1.0] * len(mc2_list)  # Default distances if embeddings are not compatible