from .operation import Operation


class SameSentence(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, **kwargs):
        return text