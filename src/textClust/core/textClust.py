from textClust.data.stream import Stream

class textClust:
    def __init__(self, data_loader:Stream, preprocessor, time_manager, micro_clusterer, macro_clusterer, history_manager):
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
            
    
    def process_one(self, point):
        point = self.preprocessor(point) if self.preprocessor is not None else point
        info = self.micro_clusterer.update(point, time=self.time_manager)
        self.history_manager.update_micro(info, time=self.time_manager)
    

    def clean_up(self):
        info = self.micro_clusterer.clean(time = self.time_manager)
        self.history_manager.clean_micro(info, time=self.time_manager)
        
    
    def cluster(self):
        self.clean_up()
        micro_clusters = self.micro_clusterer.get_clusters(time=self.time_manager)
        macro_clusters = self.macro_clusterer.cluster(micro_clusters, time=self.time_manager)
        results = self.macro_clusterer.evaluate(micro_clusters, time=self.time_manager)
        
        if self.macro_clusterer.display is not None:
            self.macro_clusterer.display(macro_clusters, micro_clusters, results, time=self.time_manager)
        
    
    def end(self):
        self.cluster()