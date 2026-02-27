from textClust.data.point import RawPoint
from textClust.data.stream import Stream
from textClust.preprocessing.preprocessor import Preprocessor
from textClust.online.clusterer import Microclusterer
from textClust.time.manager import TimeManager
from textClust.history.manager import HistoryManager
from textClust.offline.clusterer import MacroClusterer


class textClust:
    def __init__(self, data_loader:Stream, preprocessor:Preprocessor, time_manager: TimeManager, micro_clusterer:Microclusterer, macro_clusterer:MacroClusterer, history_manager: HistoryManager):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.time_manager = time_manager
        self.micro_clusterer = micro_clusterer
        self.macro_clusterer = macro_clusterer
        self.history_manager = history_manager
    
    def start(self):
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
        self.end()
            
    
    def process_one(self, point: RawPoint):
        point = self.preprocessor(point) if self.preprocessor is not None else point
        time = self.time_manager.get_current_time()
        point.timestamp = time
        clusterId = self.micro_clusterer.update(point.to_microcluster(), current_time=time)
        self.history_manager.update_text_assignment(time, point, clusterId)
    

    def clean_up(self):
        time = self.time_manager.get_current_time()
        expired_cluster_ids, merged_clusters = self.micro_clusterer.clean_up(current_time=time)
        self.history_manager.update_history(current_time=time, expired_info=expired_cluster_ids, merged_info=merged_clusters)
        self.history_manager.snapshot(current_time=time, microclusters=self.micro_clusterer.get_clusters())
        
    
    def cluster(self):
        self.clean_up()
        micro_clusters = self.micro_clusterer.get_clusters()
        macro_clusters = self.macro_clusterer.cluster(micro_clusters, current_time=self.time_manager.get_current_time())
        results = self.macro_clusterer.evaluate(macro_clusters, history_manager=self.history_manager, current_time=self.time_manager.get_current_time())
        
        if self.macro_clusterer.display is not None:
            self.macro_clusterer.display(results)
        
    def end(self):
        self.cluster()