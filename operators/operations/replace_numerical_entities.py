from .operation import Operation
import numpy as np
import wordtodigits
import re


class ReplaceNumericalEntities(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, **kwargs):
        possibilities = ["some", "a few", "many", "a lot of", ""]
        replacement = np.random.choice(possibilities)
        text = wordtodigits.convert(text)
        numbers = list(set(re.findall("\d+\.?\d*", text)))
        numbers = np.random.choice(numbers, np.random.randint(1, min(len(numbers)+1, 3)), replace=False)
        for number in numbers:
            text = re.sub(number, " "+replacement+" ", text)
        text = re.sub("\s+", " ", text)
        return text.strip()