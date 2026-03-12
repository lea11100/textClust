from textClust.online.clusterer import Microcluster

import torch

class MacroCluster(Microcluster):
    """Macro-level cluster representation built from microclusters."""

    def __init__(self, id, current_time: float, n: int, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        
class TfIdfMacroCluster(MacroCluster):
    """Macro-level cluster representation with TF-IDF summary."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.tf = tf
        
class EmbeddingMacroCluster(MacroCluster):
    """Macro-level cluster representation with embedding summary."""

    def __init__(self, id, current_time: float, n: int, embedding:torch.Tensor, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.embedding = embedding
        
class TfIdfEmbeddingMacroCluster(MacroCluster):
    """Macro-level cluster representation with both TF-IDF and embedding summaries."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, embedding:torch.Tensor, weight = 1.0):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.tf = tf
        self.embedding = embedding
        