"""Abstract stream interface for yielding raw points."""

from typing import Any, Generator
from textClust.data.point import RawPoint

class Stream:
    """Base class for data stream adapters."""

    def __init__(self, **kwargs):
        """Initialize a stream adapter."""
        pass
    
    def load(self, **kwargs) -> Generator[RawPoint | None, None, None]:
        """Yield `RawPoint` objects and a terminal `None` sentinel."""
        raise NotImplementedError("Stream.load() must be implemented by subclass")