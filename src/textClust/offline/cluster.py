from typing import Optional

from textClust.online.clusterer import Microcluster

import torch

class Macrocluster(Microcluster):
    """Macro-level cluster representation built from microclusters."""

    def __init__(self, id, current_time: float, n: int, weight = 1.0, label: Optional[str] = None):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight)
        self.label = label

class TfIdfMacrocluster(Macrocluster):
    """Macro-level cluster representation with TF-IDF summary."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, weight = 1.0, label: Optional[str] = None):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight, label)
        self.tf = tf
        
class EmbeddingMacrocluster(Macrocluster):
    """Macro-level cluster representation with embedding summary."""

    def __init__(self, id, current_time: float, n: int, embedding:torch.Tensor, weight = 1.0, label: Optional[str] = None):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight, label)
        self.embedding = embedding
        
class TfIdfEmbeddingMacrocluster(Macrocluster):
    """Macro-level cluster representation with both TF-IDF and embedding summaries."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, embedding:torch.Tensor, weight = 1.0, label: Optional[str] = None):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, weight, label)
        self.tf = tf
        self.embedding = embedding
        