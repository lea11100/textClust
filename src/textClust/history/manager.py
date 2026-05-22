"""History tracking utilities for assignments and cluster lifecycle events."""

import time

from textClust.data.point import RawPoint, Point
from textClust.online.cluster import Microcluster
from textClust.online.clusterer import Microclusterer
from textClust.preprocessing.preprocessor import TFIDFPreprocessor
from textClust.offline.cluster import Macrocluster

from typing import List

class Snapshot:
    """Immutable-like snapshot of microcluster ids and weights at a time."""

    def __init__(self, microclusters: List[Microcluster], time: float):
        """Create a snapshot from current microclusters and timestamp."""
        self.microcluster_ids = [mc.id for mc in microclusters]
        self.microcluster_weights = [mc.weight for mc in microclusters]
        self.time = time
        
class MacroSnapshot:
    """Immutable-like snapshot of macrocluster ids and weights at a time."""

    def __init__(self, macroclusters: List[Macrocluster], time: float):
        """Create a snapshot from current macroclusters and timestamp."""
        self.macrocluster_ids = [mc.id for mc in macroclusters]
        self.macrocluster_weights = [mc.weight for mc in macroclusters]
        self.macrocluster_labels = [mc.label for mc in macroclusters]
        self.time = time

class HistoryManager:
    """Store assignment history, snapshots, expiration, and merge events."""

    def __init__(self):
        """Initialize in-memory history containers."""
        self.initial_text_assignments: List[tuple] = []   # time, point_id, cluster_id
        self.true_assignments: List[tuple] = []   # point_id, true_cluster_id
        self.snapshots: List[Snapshot] = []  # List of Snapshot objects, ordered by time
        self.snapshots_macro: List[MacroSnapshot] = []  # List of MacroSnapshot objects, ordered by time
        self.cluster_expiration_time: List[tuple] = []  # time, cluster_id
        self.cluster_merge_history: List[tuple] = []  # time, merged_cluster_id, destination_cluster_ids
        self.texts = {}  # Optional: store text content for reference, keyed by text_id   
        self.threshold_history: List[tuple] = []  # time, threshold_value (if applicable)
        self.preprocessor =  TFIDFPreprocessor()  
        
        self.walltime_at_clustering = [time.time()]
        self.last_eval_time: float | None = None
        self.eval_history = [] 
    
    def update_text_assignment(self, current_time: float, point: RawPoint, cluster_id: int):
        """Record predicted assignment and optional ground-truth label."""
        self.initial_text_assignments.append((current_time, point.text_id, cluster_id))
        if point.cluster_id is not None:
            self.true_assignments.append((point.text_id, point.cluster_id))
            
        if isinstance(point, Point):
            self.texts[point.text_id] = self.preprocessor.preprocess_text(point.text)
        
    def snapshot_macro(self, current_time: float, macro_clusters: List[Macrocluster]):
        """Create a snapshot of macrocluster state at the given time."""
        self.snapshots_macro.append(MacroSnapshot(macro_clusters, time=current_time))
        
    def snapshot(self, current_time: float, clusterer: Microclusterer):
        """Append a snapshot of the current microcluster state."""
        self.snapshots.append(Snapshot(list(clusterer.get_clusters()), time=current_time))
        if hasattr(clusterer, "last_threshold"):
            attr = getattr(clusterer, "last_threshold")
            if attr is not None:
                self.threshold_history.append((current_time, attr))
        
    def update_history(self, current_time: float, expired_info: list[int], merged_info: dict[int, int]):
        """Record expired and merged cluster events at the given time."""
        for cluster_id in expired_info:
            self.cluster_expiration_time.append((current_time, cluster_id))
            
        for merged_cluster_id, destination_cluster_ids in merged_info.items():
            if merged_cluster_id != destination_cluster_ids:
                self.cluster_merge_history.append((current_time, merged_cluster_id, destination_cluster_ids))
            
    def get_final_assignments(self) -> List[tuple[int, int]]:
        """Derive final text-to-cluster assignments after merge history replay."""
        assignments = { t_id : c_id for _, t_id, c_id in self.initial_text_assignments }
        rev_assignments = {}
        for t_id, c_id in assignments.items():
            if c_id not in rev_assignments:
                rev_assignments[c_id] = []
            rev_assignments[c_id].append(t_id)
        for _, old_id, new_id in self.cluster_merge_history:
            if old_id in rev_assignments:
                if new_id not in rev_assignments:
                    rev_assignments[new_id] = []
                rev_assignments[new_id].extend(rev_assignments[old_id])
                del rev_assignments[old_id]
                
        final_assignments = []
        for c_id, t_ids in rev_assignments.items():
            for t_id in t_ids:
                final_assignments.append((t_id, c_id))
        return final_assignments
            
            