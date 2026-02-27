from typing import Optional


class TimeManager:
    def __init__(self, *args, **kwargs):
        self.current_time = 0.0
        
    def update(self, timestamp:Optional[float] = None):
        if timestamp is None:
            self.current_time += 1.0
        else:
            if timestamp < self.current_time:
                raise ValueError("Timestamps must be non-decreasing")
            self.current_time = timestamp
        
    def get_current_time(self):
        return self.current_time    
    
    def time_to_clean(self) -> bool:
        raise NotImplementedError("time_to_clean() must be implemented by subclass")
        
    def time_to_cluster(self) -> bool:
        raise NotImplementedError("time_to_cluster() must be implemented by subclass")
    
    
class FixedIntervalTimeManager(TimeManager):
    def __init__(self, clean_interval=60, cluster_interval=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clean_interval = clean_interval
        self.cluster_interval = cluster_interval 
    
    def time_to_clean(self) -> bool:
        return self.current_time % self.clean_interval == 0
    
    def time_to_cluster(self) -> bool:
        return self.current_time % self.cluster_interval == 0