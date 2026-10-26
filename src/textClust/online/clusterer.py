"""Online micro-clustering algorithms and base interfaces."""

from typing import List, Optional, Sequence, Tuple
import numpy as np
import math

import warnings

from textClust.online.cluster import Microcluster
from textClust.online.fade import DampedFade, ExpFade
from textClust.online.merge import Merger
from textClust.online.metrics import DistanceMetric

from textClust.offline.cluster import Macrocluster

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
    
    def update_with_merge_history(self, merge_history: Optional[dict[int, int]], current_time: float, *args, **kwargs):
        """Optional hook to update microclusterer state with macrocluster info."""
        pass
    
class ThresholdMicroclusterer(Microclusterer):
    """Online microclusterer based on pairwise distances using adaptive or static thresholding."""

    def __init__(self, distance_metric: DistanceMetric, fader: DampedFade, merger: Merger, adaptive_threshold=True, sigma: Optional[float] = 1.0, 
                 radius: Optional[float] = None, min_num_mc=5, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__()
        self.microclusters: List[Microcluster] = []
        self.distance_metric = distance_metric
        self.fader = fader
        self.merger = merger
        self.adaptive_threshold = adaptive_threshold
        
        if adaptive_threshold:
            assert sigma is not None, "Sigma must be provided when adaptive_threshold is True"
            self.sigma = sigma
            if radius is not None:
                warnings.warn("Warning: Both adaptive_threshold and radius are set. Radius will be ignored in favor of adaptive thresholding.")
        else:
            assert radius is not None, "Radius must be provided when adaptive_threshold is False"
            self.radius = radius
            if threshold_momentum is not None:
                warnings.warn("Warning: threshold_momentum has no effect when adaptive_threshold is False.")
            
        self.num_merged_obs = 0
        self.outlier_dist_sum = 0.0
        self.outlier_dist_squaresum = 0.0
        self.dist_sum = 0.0
        
        self.min_num_mc = min_num_mc
    
        self.threshold_momentum = threshold_momentum
        self.last_threshold = 0.0
    
    def get_clusters(self) -> Sequence[Microcluster]:
        """Return currently tracked microclusters."""
        return self.microclusters
    
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, Optional[int]]:
        """Assign or merge incoming microcluster using adaptive thresholding."""
        if self.min_num_mc > len(self.microclusters): # Maintain multiple initial clusters to get better distance stats before merging
            self.microclusters.append(mc)
            self.outlier_dist_sum += 1.0  # Treat as distance of 1 for the first few points
            self.outlier_dist_squaresum += 1.0
            return mc, None
        distances = self.distance_metric(mc, self.microclusters, *args, **kwargs)   
        if len(distances) == 0:
            self.microclusters.append(mc)
            self.outlier_dist_sum += 1.0  # Treat as distance of 1 for the first point
            self.outlier_dist_squaresum += 1.0
            return mc, None
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        distances.pop(min_dist_idx)
        avg_dist = np.average(distances) if distances else 0.0
        # Sample standard deviation (ddof=1) is undefined for fewer than two values.
        std_dist = np.std(distances, ddof=1) if len(distances) > 1 else np.nan
        
        if self.adaptive_threshold:
            threshold = avg_dist - self.sigma * std_dist
            if self.threshold_momentum and self.last_threshold > 0:
                threshold = self.threshold_momentum * self.last_threshold + (1 - self.threshold_momentum) * threshold
        else:
            threshold = self.radius
        self.last_threshold = threshold
        if min_dist < threshold and avg_dist < 1:
            closest_mc = self.microclusters[min_dist_idx]
            if self.fader is not None:
                closest_mc, _, _ = self.fader(closest_mc, current_time)
                mc, _, _ = self.fader(mc, current_time)
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
    
    def _get_expried(self, current_time: float) -> List[Tuple[Microcluster, bool, List[str]]]:
        """Fade clusters and report their expiration status without deleting anything."""
        return [self.fader(mc, current_time) for mc in self.microclusters]

    def _delete_expired(self, current_time: float) -> List[int]:
        """Remove expired terms and clusters, returning expired cluster ids."""
        expired_cluster_ids = []
        active_clusters = []
        for mc, is_expired, expired_terms in self._get_expried(current_time):
            if is_expired:
                expired_cluster_ids.append(mc.id)
                continue
            if expired_terms:
                assert isinstance(mc.embedding, dict), "Expected term-fading microcluster to have a dictionary embedding"
                for term in expired_terms:
                    del mc.embedding[term]
            active_clusters.append(mc)
        self.microclusters = active_clusters
        return expired_cluster_ids
        
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List[int], dict[int, int]]:
        """Run expiration pass and merge close clusters.

        Returns:
            A tuple of `(expired_cluster_ids, merge_history)`.
        """        
        expired_cluster_ids =self._delete_expired(current_time)
        merge_history = self._merge_clusters(current_time, *args, **kwargs)
        return expired_cluster_ids, merge_history
    
    def _merge_clusters(self, current_time: float, *args, **kwargs):
        merge_history = {}
        if self.adaptive_threshold:
            threshold = self.dist_sum / self.num_merged_obs if self.num_merged_obs > 0 else 0.0
        else:
            threshold = self.radius
        self.num_merged_obs = 0
        self.dist_sum = 0.0
        removed = np.zeros(len(self.microclusters), dtype=bool)
        for i in range(len(self.microclusters)-1):
            self.microclusters[i], _, _ = self.fader(self.microclusters[i], current_time)
            if removed[i]:
                continue
            m_dist = self.distance_metric(self.microclusters[i], self.microclusters[i+1:], *args, **kwargs)
            for j, dist in enumerate(m_dist, start=i+1):
                if removed[j]:
                    continue
                
                # Merge if below threshold
                if dist < threshold:
                    self.microclusters[i], _, _ = self.fader(self.microclusters[i], current_time)
                    self.microclusters[j], _, _ = self.fader(self.microclusters[j], current_time)
                    self.microclusters[i] = self.merger(self.microclusters[i], self.microclusters[j], current_time=current_time)
                    removed[j] = True
                    merge_history[self.microclusters[j].id] = self.microclusters[i].id
                    
                    
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return merge_history
        
    def update_with_merge_history(self, merge_history: Optional[dict[int, int]], current_time: float, *args, **kwargs):
        if merge_history is None or len(merge_history) == 0:
            return
        mc_dict = {mc.id: mc for mc in self.microclusters}
        for dest_id in set(merge_history.values()):
            for c_id in [mid for mid, dest in merge_history.items() if dest == dest_id]:
                if dest_id == c_id:
                    continue
                if dest_id != -1: # If dest_id is -1, it means the cluster should be deleted, so we skip merging and just mark for deletion 
                    mc_dict[dest_id], _, _ = self.fader(mc_dict[dest_id], current_time)
                    mc_dict[c_id], _, _ = self.fader(mc_dict[c_id], current_time)
                    mc_dict[dest_id] = self.merger(mc_dict[dest_id], mc_dict[c_id], current_time=current_time)
                del mc_dict[c_id]
        self.microclusters = list(mc_dict.values())
    
class TfIdfThresholdMicroclusterer(ThresholdMicroclusterer):
    """Online microclusterer based on pairwise distances using adaptive or static thresholding, with TF-IDF vector representation."""

    def __init__(self, distance_metric: DistanceMetric, fader: DampedFade, merger: Merger, 
                 adaptive_threshold=True, sigma: Optional[float] = 1.0, radius: Optional[float] = None,
                 min_num_mc=5, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, sigma=sigma, min_num_mc=min_num_mc,
                        adaptive_threshold=adaptive_threshold, radius=radius,
                        threshold_momentum=threshold_momentum, *args, **kwargs)
    
    def _update(self, mc: Microcluster, current_time: float, *args, **kwargs) -> Tuple[Microcluster, Optional[int]]:
        """Assign or merge incoming microcluster using adaptive thresholding."""
        idf = self._get_idf()
        return super()._update(mc, current_time, idf=idf, n=len(self.microclusters), *args, **kwargs)
    
    def clean_up(self, current_time: float, *args, **kwargs) -> tuple[List[int], dict[int, int]]:
        """Run expiration pass and merge close clusters.

        Returns:
            A tuple of `(expired_cluster_ids, merge_history)`.
        """
        expired_cluster_ids = self._delete_expired(current_time)
        idf = self._get_idf()
        merge_history = self._merge_clusters(current_time, idf=idf, n=len(self.microclusters), *args, **kwargs)
        return expired_cluster_ids, merge_history
       
    def _get_idf(self) -> dict[str, float]:
        """Compute per-term inverse document frequency over active clusters."""
        result = {}
        
        # Count document frequency for each term
        for mc in self.microclusters:
            assert isinstance(mc.embedding, dict), "Expected TF-IDF microcluster to have embedding as a dictionary"
            for term in mc.embedding.keys():
                if term not in result:
                    result[term] = 1
                else:
                    result[term] += 1
        
        # Calculate IDF: 1 + log(N / df)
        num_clusters = len(self.microclusters)
        for term in result.keys():
            result[term] = 1 + math.log(num_clusters / result[term])
        
        return result
    
class EmbeddingThresholdClusterer(ThresholdMicroclusterer):
    """Online microclusterer based on pairwise distances using adaptive or static thresholding, with LLM embedding representation."""
    def __init__(self,
                 distance_metric: DistanceMetric, fader: DampedFade, merger: Merger,
                 adaptive_threshold=True, sigma: Optional[float] = 1.0, radius: Optional[float] = None,
                 min_num_mc=15, threshold_momentum:Optional[float] = None, *args, **kwargs):
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, sigma=sigma, min_num_mc=min_num_mc,
                         adaptive_threshold=adaptive_threshold, radius=radius,
                         threshold_momentum=threshold_momentum, *args, **kwargs)