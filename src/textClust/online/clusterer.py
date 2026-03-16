"""Online micro-clustering algorithms and base interfaces."""

from typing import List, Optional, Sequence, Tuple
import numpy as np
import math

from textClust.online.cluster import EmbeddingMicrocluster, Microcluster, TfIdfMicrocluster
from textClust.online.fade import DampedFade, EmbeddingDampedFade, EmbeddingDampedFade, TfIdfDampedFade
from textClust.online.merge import EmbeddingMerger, EmbeddingMerger, Merger, TfIdfMerger
from textClust.online.metrics import DistanceMetric, EmbeddingDistanceMetric, TfIdfDistanceMetric

from textClust.offline.cluster import MacroCluster

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
    
class AdaptiveThresholdMicroclusterer(Microclusterer):
    """Adaptive-threshold online microclusterer based on pairwise distances."""

    def __init__(self, distance_metric: DistanceMetric, fader: DampedFade, merger: Merger, sigma=1.0,
                 min_num_mc=5, include_min_for_threshold=False, threshold_momentum=None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__()
        self.microclusters: List[Microcluster] = []
        self.sigma = sigma
        self.distance_metric = distance_metric
        self.fader = fader
        self.merger = merger
        
        self.num_merged_obs = 0
        self.outlier_dist_sum = 0.0
        self.outlier_dist_squaresum = 0.0
        self.dist_sum = 0.0
        
        self.min_num_mc = min_num_mc
        
        self.include_min_for_threshold = include_min_for_threshold
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
        
        idf = self._get_idf() if self.distance_metric.requires_idf else None
        if self.distance_metric.support_batch:
            batch_fn = getattr(self.distance_metric, "batch_distance", None) 
            if batch_fn is not None:
                distances = batch_fn(mc, self.microclusters, idf=idf, *args, **kwargs)
            else:
                raise ValueError("Distance metric claims to support batch distance but does not implement batch_distance()")
        else:
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
        if self.threshold_momentum and self.last_threshold > 0:
            threshold = self.threshold_momentum * self.last_threshold + (1 - self.threshold_momentum) * threshold
        self.last_threshold = threshold
        if min_dist < threshold and avg_dist < 1:
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
                    self.microclusters[i] = self.merger(self.microclusters[i], self.microclusters[j], current_time=current_time, fader=self.fader)
                    removed[j] = True
                    merge_history[self.microclusters[j].id] = self.microclusters[i].id
                    
                    
        self.microclusters = [c for r, c in zip(removed, self.microclusters) if not r]
        return expired_cluster_ids, merge_history
        
    def _get_idf(self, additional_mc: Optional[TfIdfMicrocluster] = None) -> dict[str, float]:
        """Compute per-term inverse document frequency over active clusters."""
        raise NotImplementedError("IDF calculation is only applicable for TF-IDF microclusters and must be implemented by subclass if needed.")
    
    def update_with_merge_history(self, merge_history: Optional[dict[int, int]], current_time: float, *args, **kwargs):
        if merge_history is None or len(merge_history) == 0:
            return
        mc_dict = {mc.id: mc for mc in self.microclusters}
        for dest_id in set(merge_history.values()):
            for c_id in [mid for mid, dest in merge_history.items() if dest == dest_id]:
                if dest_id == c_id:
                    continue
                if dest_id != -1: # If dest_id is -1, it means the cluster should be deleted, so we skip merging and just mark for deletion 
                    mc_dict[dest_id] = self.merger(mc_dict[dest_id], mc_dict[c_id], current_time=current_time, fader=self.fader)
                del mc_dict[c_id]
        self.microclusters = list(mc_dict.values())
    
class TfIdfAdaptiveThresholdMicroclusterer(AdaptiveThresholdMicroclusterer):
    """Adaptive-threshold online microclusterer based on pairwise distances."""

    def __init__(self, distance_metric: TfIdfDistanceMetric, fader: TfIdfDampedFade, merger: TfIdfMerger, sigma=1.0,
                 min_num_mc=5, include_min_for_threshold=False, threshold_momentum: Optional[float] = None, *args, **kwargs):
        """Initialize adaptive threshold microclusterer parameters."""
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, sigma=sigma, min_num_mc=min_num_mc,
                         include_min_for_threshold=include_min_for_threshold, threshold_momentum=threshold_momentum, *args, **kwargs)
        self.microclusters: List[TfIdfMicrocluster] = [] # type: ignore
        
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
    
class EmbeddingAdaptiveThresholdClusterer(AdaptiveThresholdMicroclusterer):
    def __init__(self,
                 distance_metric: EmbeddingDistanceMetric, fader: EmbeddingDampedFade, merger: EmbeddingMerger, sigma=1.0, 
                 min_num_mc=15, include_min_for_threshold=True, threshold_momentum:Optional[float] = None, *args, **kwargs):
        super().__init__(distance_metric=distance_metric, fader=fader, merger=merger, sigma=sigma, min_num_mc=min_num_mc,
                         include_min_for_threshold=include_min_for_threshold, threshold_momentum=threshold_momentum, *args, **kwargs)