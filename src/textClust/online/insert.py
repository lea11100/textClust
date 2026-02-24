class Inserter:
    def __init__(self, **kwargs):
        pass
    
    def insert(self, microclusters, point, **kwargs):
        raise NotImplementedError("Inserter.insert() must be implemented by subclass")
    
class SimpleInserter(Inserter):
    pass