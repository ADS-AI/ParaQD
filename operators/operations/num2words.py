from num2words import num2words
from .operation import Operation
import re
from nltk.tokenize import sent_tokenize


class Num2Words(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, **kwargs):
        numbers = list(set(re.findall("\d+\.?\d*", text)))
        mapping = {number: num2words(number) for number in numbers}
        for number in mapping:
            text = text.replace(number, mapping[number])
        text = re.sub("\s+", " ", text)
        sentences = sent_tokenize(text)
        text = " ".join(sentence[0].upper() + sentence[1:] for sentence in sentences)
        return text.strip()