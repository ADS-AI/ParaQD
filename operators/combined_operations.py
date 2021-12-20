from abc import ABC, abstractmethod


class CombinedOperation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, text, ops=[], **kwargs):
        pass