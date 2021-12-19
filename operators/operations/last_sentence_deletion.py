from .operation import Operation
from nltk.tokenize import sent_tokenize


class DeleteLastSentence(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, **kwargs):
        text_split = sent_tokenize(text)
        if len(text_split) > 1:
            text = " ".join(text_split[:-1])
            return text
        return " ".join(text.split()[:-4]) # Deletes last 4 tokens if there is only 1 sentence