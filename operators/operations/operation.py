from abc import ABC, abstractmethod

class Operation(ABC):
    """
    Abstract class for augmenting a given text.
    """

    @abstractmethod
    def generate(self, text, **kwargs):
        """
        Corrupts the given text.
        """
        pass