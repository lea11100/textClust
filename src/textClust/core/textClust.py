"""Core orchestration class for streaming text clustering.

This module wires together the input stream, preprocessing, micro-clustering,
macro-clustering, temporal management, and history tracking components.
"""

from textClust.data.point import RawPoint
from textClust.data.stream import Stream
from textClust.preprocessing.preprocessor import Preprocessor
from textClust.online.clusterer import Microclusterer
from textClust.time.manager import TimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import MacroClusterer


class textClust:
    """Run the text clustering pipeline over a data stream.

    The class coordinates data loading, preprocessing, time progression,
    micro-cluster updates, periodic cleanup, macro-clustering, and result
    reporting through the supplied collaborator objects.
    """

    def __init__(self, data_loader:Stream, preprocessor:Preprocessor, time_manager: TimeManager, 
                 micro_clusterer:Microclusterer, macro_clusterer:MacroClusterer, history_manager: HistoryManager,
                 verbose: bool = True):
        """Initialize the text clustering pipeline.

        Args:
            data_loader: Stream provider for incoming raw text points.
            preprocessor: Optional callable that transforms raw points.
            time_manager: Time controller for clustering cadence and cleanup.
            micro_clusterer: Online micro-clustering component.
            macro_clusterer: Offline macro-clustering and evaluation component.
            history_manager: Tracks assignments and cluster lifecycle history.
            verbose: Whether to call display callbacks when available.
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.time_manager = time_manager
        self.micro_clusterer = micro_clusterer
        self.macro_clusterer = macro_clusterer
        self.history_manager = history_manager
        self.verbose = verbose
        
    def start(self):
        """Start processing the input stream until exhaustion.

        For each incoming point, this updates time, assigns the point to a
        micro-cluster, and triggers periodic cleanup and macro-clustering
        according to the configured `TimeManager` policy.
        """
        stream = self.data_loader.load()
        while True:
            point = next(stream)
            if point is None:
                break
            self.time_manager.update(point.timestamp)
            self.process_one(point)
            if self.time_manager.time_to_clean():
                self.clean_up()
            if self.time_manager.time_to_cluster():
                self.cluster()
            if self.time_manager.time_to_log():
                self.log()
            
        self.end()
            
    
    def process_one(self, point: RawPoint):
        """Process a single raw point through preprocessing and assignment.

        Args:
            point: Incoming raw data point.
        """
        point = self.preprocessor(point) if self.preprocessor is not None else point
        time = self.time_manager.get_current_time()
        point.timestamp = time
        clusterId = self.micro_clusterer.update(point.to_microcluster(), current_time=time)
        self.history_manager.update_text_assignment(time, point, clusterId)
    
    def log(self):
        """Log intermediate results or diagnostics if enabled."""
        time = self.time_manager.get_current_time()
        self.history_manager.snapshot(current_time=time, clusterer=self.micro_clusterer)

    def clean_up(self):
        """Run micro-cluster cleanup and persist history snapshots.

        This removes expired clusters, records merge/expiration metadata, and
        stores a snapshot of current micro-cluster state.
        """
        time = self.time_manager.get_current_time()
        expired_cluster_ids, merged_clusters = self.micro_clusterer.clean_up(current_time=time)
        self.history_manager.update_history(current_time=time, expired_info=expired_cluster_ids, merged_info=merged_clusters)
        
    
    def cluster(self):
        """Perform macro-clustering on current micro-clusters and evaluate.

        If configured, intermediate results are displayed.
        """
        micro_clusters = list(self.micro_clusterer.get_clusters())
        macro_clusters, merge_history = self.macro_clusterer.cluster(micro_clusters, self.history_manager, current_time=self.time_manager.get_current_time())
        self.micro_clusterer.update_with_merge_history(merge_history, current_time=self.time_manager.get_current_time())
        if merge_history is not None:
            self.history_manager.update_history(current_time=self.time_manager.get_current_time(), expired_info=[], merged_info=merge_history)
        self.history_manager.snapshot_macro(current_time=self.time_manager.get_current_time(), macro_clusters=macro_clusters)
        results = self.macro_clusterer.evaluate(macro_clusters, self.history_manager, current_time=self.time_manager.get_current_time())
        
        if self.macro_clusterer.display is not None and self.verbose:
            self.macro_clusterer.display(results)
        
    def end(self):
        """Finalize processing and emit final visualization/report if enabled."""
        self.cluster()
        if self.macro_clusterer.display_final is not None and self.verbose:
            self.macro_clusterer.display_final(micro_clusters=list(self.micro_clusterer.get_clusters()), history_manager=self.history_manager, current_time=self.time_manager.get_current_time())