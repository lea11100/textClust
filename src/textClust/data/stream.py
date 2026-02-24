from typing import Any, Generator
from textClust.data.point import RawPoint

class Stream:
    def __init__(self, **kwargs):
        pass
    
    def load(self, **kwargs) -> Generator[RawPoint | None, None, None]:
        raise NotImplementedError("Stream.load() must be implemented by subclass")