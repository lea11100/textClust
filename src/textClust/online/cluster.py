"""Microcluster data structures for online clustering."""

class Microcluster:
    """Compact summary of a group of text points."""

    def __init__(self, id, tf:dict[str, float], current_time: float, n: int, embedding=None, weight = 1.0):
        """Initialize a microcluster.

        Args:
            id: Cluster identifier.
            tf: Term-frequency map for represented points.
            current_time: Last update timestamp.
            n: Number of aggregated points.
            embedding: Optional embedding summary.
            weight: Current cluster weight.
        """
        self.id = id
        self.tf = tf
        self.last_update_time = current_time
        self.embedding = embedding
        self.weight = weight
        self.n = n
                
    def __eq__(self, value):
        """Return value-based equality for testing and comparisons."""
        if not isinstance(value, Microcluster):
            return False
        return (self.id == value.id and
                self.tf == value.tf and
                self.weight == value.weight and
                self.last_update_time == value.last_update_time and
                self.n == value.n and
                (self.embedding == value.embedding if self.embedding is not None and value.embedding is not None else self.embedding is value.embedding))