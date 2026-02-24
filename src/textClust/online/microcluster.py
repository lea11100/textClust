class Microcluster:
    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, embedding=None, weight = 1.0):
        self.id = id
        self.tf = tf
        self.last_update_time = current_time
        self.embedding = embedding
        self.weight = weight
        self.n = n
                
    def __eq__(self, value):
        if not isinstance(value, Microcluster):
            return False
        return (self.id == value.id and
                self.tf == value.tf and
                self.weight == value.weight and
                self.last_update_time == value.last_update_time and
                self.n == value.n and
                (self.embedding == value.embedding if self.embedding is not None and value.embedding is not None else self.embedding is value.embedding))

class Microclusterer:
    def __init__(self):
        pass
    
    def get_microclusters(self) -> list[Microcluster]: #TODO
        raise NotImplementedError("get_microclusters() must be implemented by subclass")
    
class SimpleMicroclusterer(Microclusterer):
    def __init__(self):
        super().__init__()
        self.microclusters = []
    
    def get_microclusters(self) -> list[Microcluster]:
        return self.microclusters