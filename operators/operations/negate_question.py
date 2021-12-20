from .operation import Operation


class NegateQuestion(Operation):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, text, **kwargs):
        raise NotImplementedError