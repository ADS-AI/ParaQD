from .combined_operations import CombinedOperation
from .operations import BackTranslate, UnitExpansion, SameSentence, Num2Words

class PositiveSamples(CombinedOperation):
    def __init__(self):
        self.operations = {
            "BackTranslate": BackTranslate(),
            "SameSentence": SameSentence(),
            "Num2Words": Num2Words(),
            "UnitExpansion": UnitExpansion()
        }

    def generate(self, text, ops=["BackTranslate", "SameSentence", "Num2Words", "UnitExpansion"]):
        positives = []
        for op in ops:
            if op not in self.operations: continue
            operator = self.operations[op]
            positives.append(operator.generate(text))
        return positives