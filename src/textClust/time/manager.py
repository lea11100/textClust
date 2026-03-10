"""Time management primitives controlling cleanup and clustering cadence."""

from typing import Optional


class TimeManager:
    """Base clock manager for streaming updates."""

    def __init__(self, *args, **kwargs):
        """Initialize the internal clock at zero."""
        self.current_time = 0.0
        self.time_repeated = False # Flag to track if time has been repeated 
        
    def update(self, timestamp:Optional[float] = None):
        """Advance time using explicit timestamp or unit step."""
        if timestamp is None:
            self.current_time += 1.0
        else:
            if timestamp < self.current_time:
                raise ValueError("Timestamps must be non-decreasing")
            if timestamp == self.current_time:
                self.time_repeated = True
            else:
                self.time_repeated = False
            self.current_time = timestamp
        
    def get_current_time(self):
        """Return the current logical time."""
        return self.current_time    
    
    def time_to_clean(self) -> bool:
        """Return whether maintenance should run at current time."""
        raise NotImplementedError("time_to_clean() must be implemented by subclass")
        
    def time_to_cluster(self) -> bool:
        """Return whether macro-clustering should run at current time."""
        raise NotImplementedError("time_to_cluster() must be implemented by subclass")
    
    def time_to_log(self) -> bool:
        """Return whether logging should run at current time."""
        return False  # Optional logging, default to never
    
    
class FixedIntervalTimeManager(TimeManager):
    """Time manager that triggers actions on fixed time intervals."""

    def __init__(self, clean_interval=60, cluster_interval=300, log_interval=60, *args, **kwargs):
        """Initialize fixed cleanup and clustering intervals."""
        super().__init__(*args, **kwargs)
        self.clean_interval = clean_interval
        self.cluster_interval = cluster_interval
        self.log_interval = log_interval

    def time_to_log(self) -> bool:
        """Return true when current time hits a logging interval."""
        return self.current_time % self.log_interval == 0 and not self.time_repeated  # Avoid repeated logging if time is repeated

    def time_to_clean(self) -> bool:
        """Return true when current time hits a cleanup interval."""
        return self.current_time % self.clean_interval == 0 and not self.time_repeated  # Avoid repeated cleaning if time is repeated
    
    def time_to_cluster(self) -> bool:
        """Return true when current time hits a clustering interval."""
        return self.current_time % self.cluster_interval == 0 and not self.time_repeated  # Avoid repeated clustering if time is repeated