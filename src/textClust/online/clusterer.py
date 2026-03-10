"""Online micro-clustering algorithms and base interfaces."""

from typing import List, Optional, Sequence, Tuple
import numpy as np
import math

from textClust.online.cluster import EmbeddingMicrocluster, Microcluster, TfIdfMicrocluster
from textClust.online.fade import EmbeddingDampedFade, EmbeddingDampedFade, TfIdfDampedFade
from textClust.online.merge import EmbeddingMerger, EmbeddingMerger, TfIdfMerger
from textClust.online.metrics import EmbeddingDistanceMetric, TfIdfDistanceMetric

class Microclusterer:
    """Base interface for assigning points to microclusters."""

    def __init__(self, *args, **kwargs):
        """Initialize cluster id generation state."""
        self.clusterId = 1
        
    def update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> int:
        """Update state with one microcluster candidate and return cluster id."""
        mc, clusterId = self._update(mc, current_time, *args, **kwargs)
        if clusterId is None:
            clusterId = self.pull_new_clusterId()
            mc.id = clusterId
        return clusterId
    
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, Optional[int]]:
        """Internal update hook implemented by subclasses.

        Returns:
            Tuple of (updated microcluster, existing cluster id if merged, otherwise `None`).
        """
        raise NotImplementedError("update() must be implemented by subclass")
     
    def get_clusters(self, *args, **kwargs) -> Sequence[Microcluster]:
        """Return the current list of active microclusters."""
        raise NotImplementedError("get_clusters() must be implemented by subclass")
    
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List, dict]:
        """Expire and merge stale/close clusters, returning event metadata."""
        raise NotImplementedError("clean_up() must be implemented by subclass")
    
    def pull_new_clusterId(self):
        """Generate and return a fresh cluster id."""
        self.clusterId += 1
        return self.clusterId - 1
    
class TfIdfAdaptiveThresholdMicroclusterer(Microclusterer):
    """Adaptive-threshold online microclusterer based on pairwise distances."""

    def __init__(self, distance_metric: TfIdfDistanceMetric, fader: TfIdfDampedFade, merger: TfIdfMerger, sigma=1.0, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__()
        self.microclusters = []
        self.sigma = sigma
        self.distance_metric = distance_metric
        self.fader = fader
        self.merger = merger
        
        self.num_merged_obs = 0
        self.outlier_dist_sum = 0.0
        self.outlier_dist_squaresum = 0.0
        self.dist_sum = 0.0
    
    def get_clusters(self) -> Sequence[TfIdfMicrocluster]:
        """Return currently tracked microclusters."""
        return self.microclusters
    
    def _update(self, mc: TfIdfMicrocluster, current_time: float, *args, **kwargs) -> Tuple[TfIdfMicrocluster, Optional[int]]:
        """Assign or merge incoming microcluster using adaptive thresholding."""
        idf = self._get_idf() if self.distance_metric.requires_idf else None
        distances = [self.distance_metric(mc, existing_mc, idf=idf, *args, **kwargs) for existing_mc in self.microclusters]
        if len(distances) == 0:
            self.microclusters.append(mc)
            self.outlier_dist_sum += 1.0  # Treat as distance of 1 for the first point
            self.outlier_dist_squaresum += 1.0
            return mc, None
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        distances.pop(min_dist_idx)
        avg_dist = np.average(distances) if distances else 0.0
        std_dist = np.std(distances) if distances else 0.0
        
        threshold = avg_dist - self.sigma * std_dist
        if min_dist < threshold:
            closest_mc = self.microclusters[min_dist_idx]
            merged_mc = self.merger(closest_mc, mc, current_time=current_time, fader=self.fader)
            self.microclusters[min_dist_idx] = merged_mc
            
            self.num_merged_obs += 1
            self.dist_sum += min_dist 
            
            return merged_mc, closest_mc.id
        else:
            self.microclusters.append(mc)
            self.outlier_dist_sum += min_dist
            self.outlier_dist_squaresum += min_dist ** 2
            
            return mc, None
    
    def _delete_expired(self, current_time: float):
        """Fade all clusters and remove expired ones.

        Returns:
            List of expired cluster ids.
        """
        removed = np.zeros(len(self.microclusters), dtype=bool)
        for i in range(len(self.microclusters)):
            mc, is_expired =self.fader(self.microclusters[i], current_time)
            if is_expired:
                removed[i] = True
            else:
                self.microclusters[i] = mc
        expired_cluster_ids = [self.microclusters[i].id for i in range(len(self.microclusters)) if removed[i]]
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return expired_cluster_ids
        
                
        
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List[int], dict[int, int]]:
        """Run expiration pass and merge close clusters.

        Returns:
            A tuple of `(expired_cluster_ids, merge_history)`.
        """
        threshold = self.dist_sum / self.num_merged_obs if self.num_merged_obs > 0 else 0.0
        self.num_merged_obs = 0
        self.dist_sum = 0.0
        
        expired_cluster_ids =self._delete_expired(current_time)
        merge_history = {}
        
        removed = np.zeros(len(self.microclusters), dtype=bool)
        for i in range(len(self.microclusters)):
            self.fader(self.microclusters[i], current_time)
            if removed[i]:
                continue
            for j in range(i + 1, len(self.microclusters)):
                if removed[j]:
                    continue
                
                # Calculate distance
                idf = self._get_idf() if self.distance_metric.requires_idf else None
                m_dist = self.distance_metric(self.microclusters[i], self.microclusters[j], idf=idf)
                
                # Merge if below threshold
                if m_dist < threshold:
                    self.merger(self.microclusters[i], self.microclusters[j], current_time=current_time, fader=self.fader)
                    removed[j] = True
                    merge_history[self.microclusters[j].id] = self.microclusters[i].id
                    
                    
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return expired_cluster_ids, merge_history
        
    def _get_idf(self, additional_mc: Optional[TfIdfMicrocluster] = None) -> dict[str, float]:
        """Compute per-term inverse document frequency over active clusters."""
        result = {}
        
        # Count document frequency for each term
        for mc in self.microclusters:
            for term in mc.tf.keys():
                if term not in result:
                    result[term] = 1
                else:
                    result[term] += 1
        if additional_mc is not None:
            for term in additional_mc.tf.keys():
                if term not in result:
                    result[term] = 1
                else:
                    result[term] += 1
        
        # Calculate IDF: 1 + log(N / df)
        num_clusters = len(self.microclusters)
        if additional_mc is not None:
            num_clusters += 1
        for term in result.keys():
            result[term] = 1 + math.log(num_clusters / result[term])
        
        return result
    
class EmbeddingAdaptiveThresholdClusterer(Microclusterer):
    def __init__(self,
                 distance_metric: EmbeddingDistanceMetric, fader: EmbeddingDampedFade, merger: EmbeddingMerger, sigma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.microclusters = []
        self.distance_metric = distance_metric
        self.fader = fader
        self.merger = merger
        self.sigma = sigma
        
        self.num_merged_obs = 0
        self.outlier_dist_sum = 0.0
        self.outlier_dist_squaresum = 0.0
        self.dist_sum = 0.0
        
    def _update(self, mc: EmbeddingMicrocluster, current_time: float, *args, **kwargs) -> Tuple[EmbeddingMicrocluster, Optional[int]]:
        """Assign or merge incoming microcluster using adaptive thresholding."""
        distances = [self.distance_metric(mc, existing_mc, *args, **kwargs) for existing_mc in self.microclusters]
        if len(distances) == 0:
            self.microclusters.append(mc)
            self.outlier_dist_sum += 1.0  # Treat as distance of 1 for the first point
            self.outlier_dist_squaresum += 1.0
            return mc, None
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        distances.pop(min_dist_idx)
        avg_dist = np.average(distances) if distances else 0.0
        std_dist = np.std(distances) if distances else 0.0
        
        threshold = avg_dist - self.sigma * std_dist
        if min_dist < threshold:
            closest_mc = self.microclusters[min_dist_idx]
            merged_mc = self.merger(closest_mc, mc, current_time=current_time)
            self.microclusters[min_dist_idx] = merged_mc
            
            self.num_merged_obs += 1
            self.dist_sum += min_dist 
            
            return merged_mc, closest_mc.id
        else:
            self.microclusters.append(mc)
            self.outlier_dist_sum += min_dist
            self.outlier_dist_squaresum += min_dist ** 2
            
            return mc, None
        
    def get_clusters(self) -> Sequence[EmbeddingMicrocluster]:
        """Return currently tracked microclusters."""
        return self.microclusters
    
    def _delete_expired(self, current_time: float):
        """Fade all clusters and remove expired ones.

        Returns:
            List of expired cluster ids.
        """
        removed = np.zeros(len(self.microclusters), dtype=bool)
        for i in range(len(self.microclusters)):
            mc, is_expired =self.fader(self.microclusters[i], current_time)
            if is_expired:
                removed[i] = True
            else:
                self.microclusters[i] = mc
        expired_cluster_ids = [self.microclusters[i].id for i in range(len(self.microclusters)) if removed[i]]
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return expired_cluster_ids
        
                
        
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List[int], dict[int, int]]:
        """Run expiration pass and merge close clusters.

        Returns:
            A tuple of `(expired_cluster_ids, merge_history)`.
        """
        threshold = self.dist_sum / self.num_merged_obs if self.num_merged_obs > 0 else 0.0
        self.num_merged_obs = 0
        self.dist_sum = 0.0
        
        expired_cluster_ids =self._delete_expired(current_time)
        merge_history = {}
        
        removed = np.zeros(len(self.microclusters), dtype=bool)
        for i in range(len(self.microclusters)):
            self.fader(self.microclusters[i], current_time)
            if removed[i]:
                continue
            for j in range(i + 1, len(self.microclusters)):
                if removed[j]:
                    continue
                
                # Calculate distance
                m_dist = self.distance_metric(self.microclusters[i], self.microclusters[j])
                
                # Merge if below threshold
                if m_dist < threshold:
                    self.merger(self.microclusters[i], self.microclusters[j], current_time=current_time)
                    removed[j] = True
                    merge_history[self.microclusters[j].id] = self.microclusters[i].id
                    
                    
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return expired_cluster_ids, merge_history