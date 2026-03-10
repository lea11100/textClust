"""History tracking utilities for assignments and cluster lifecycle events."""

from textClust.data.point import RawPoint, Point
from textClust.online.cluster import Microcluster
from textClust.preprocessing.preprocessor import TFIDFPreprocessor

from typing import List

class Snapshot:
    """Immutable-like snapshot of microcluster ids and weights at a time."""

    def __init__(self, microclusters: List[Microcluster], time: float):
        """Create a snapshot from current microclusters and timestamp."""
        self.microcluster_ids = [mc.id for mc in microclusters]
        self.microcluster_weights = [mc.weight for mc in microclusters]
        self.time = time

class HistoryManager:
    """Store assignment history, snapshots, expiration, and merge events."""

    def __init__(self):
        """Initialize in-memory history containers."""
        self.initial_text_assignments: List[tuple] = []   # time, point_id, cluster_id
        self.true_assignments: List[tuple] = []   # point_id, true_cluster_id
        self.snapshots: List[Snapshot] = []  # List of Snapshot objects, ordered by time
        self.cluster_expiration_time: List[tuple] = []  # time, cluster_id
        self.cluster_merge_history: List[tuple] = []  # time, merged_cluster_id, destination_cluster_ids
        self.texts = {}  # Optional: store text content for reference, keyed by text_id   
        self.preprocessor =  TFIDFPreprocessor()   
    
    def update_text_assignment(self, current_time: float, point: RawPoint, cluster_id: int):
        """Record predicted assignment and optional ground-truth label."""
        self.initial_text_assignments.append((current_time, point.text_id, cluster_id))
        if point.cluster_id is not None:
            self.true_assignments.append((point.text_id, point.cluster_id))
            
        if isinstance(point, Point):
            self.texts[point.text_id] = self.preprocessor.preprocess_text(point.text)
        
    def snapshot(self, current_time: float, microclusters: List[Microcluster]):
        """Append a snapshot of the current microcluster state."""
        self.snapshots.append(Snapshot(microclusters, time=current_time))
        
    def update_history(self, current_time: float, expired_info: list[int], merged_info: dict[int, int]):
        """Record expired and merged cluster events at the given time."""
        for cluster_id in expired_info:
            self.cluster_expiration_time.append((current_time, cluster_id))
            
        for merged_cluster_id, destination_cluster_ids in merged_info.items():
            self.cluster_merge_history.append((current_time, merged_cluster_id, destination_cluster_ids))
            
            