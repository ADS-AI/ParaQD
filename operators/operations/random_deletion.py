from .operation import Operation
import numpy as np


class RandomDeletion(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, soften=False, **kwargs):
        words = text.split()
        num = np.random.randint(1, max(4, len(words)//10))
        if soften:
            num = 1
        idxs = np.random.randint(0, len(words), size=num)
        text = " ".join([words[i] for i in range(len(words)) if i not in idxs])
        return text
