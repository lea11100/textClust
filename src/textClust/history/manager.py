from textClust.data.point import RawPoint
from textClust.online.cluster import Microcluster

from typing import List

class Snapshot:
    def __init__(self, microclusters: List[Microcluster], time: float):
        self.microcluster_ids = [mc.id for mc in microclusters]
        self.microcluster_weights = [mc.weight for mc in microclusters]
        self.time = time

class HistoryManager:
    def __init__(self):
        self.initial_text_assignments = []   # time, point_id, cluster_id
        self.true_assignments = []   # point_id, true_cluster_id
        self.snapshots = []  # List of Snapshot objects, ordered by time
        self.cluster_expiration_time = []  # time, cluster_id
        self.cluster_merge_history = []  # time, merged_cluster_id, destination_cluster_ids      
    
    def update_text_assignment(self, current_time: float, point: RawPoint, cluster_id: int):
        self.initial_text_assignments.append((current_time, point.text_id, cluster_id))
        if point.cluster_id is not None:
            self.true_assignments.append((point.text_id, point.cluster_id))
        
    def snapshot(self, current_time: float, microclusters: List[Microcluster]):
        self.snapshots.append(Snapshot(microclusters, time=current_time))
        
    def update_history(self, current_time: float, expired_info, merged_info):
        for cluster_id in expired_info:
            self.cluster_expiration_time.append((current_time, cluster_id))
            
        for merged_cluster_id, destination_cluster_ids in merged_info.items():
            self.cluster_merge_history.append((current_time, merged_cluster_id, destination_cluster_ids))
            
            