from typing import Optional

from textClust.online.clusterer import Microcluster

import torch

class Macrocluster(Microcluster):
    """Macro-level cluster representation built from microclusters."""

    def __init__(self, id, current_time: float, embedding: dict[str, float]|torch.Tensor, n: int, weight = 1.0, label: Optional[str] = None):
        """Initialize a macrocluster."""
        super().__init__(id, current_time, n, embedding, weight)
        self.label = label
        