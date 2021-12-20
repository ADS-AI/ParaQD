from .combined_operations import CombinedOperation
from operations import BackTranslate, UnitExpansion, SameSentence, Num2Words

class PositiveSamples(CombinedOperation):
    def __init__(self):
        self.operations = [
            BackTranslate(),
            SameSentence(),
            Num2Words(),
            UnitExpansion()
        ]

    def generate(self, text, ops=["BackTranslate", "SameSentence", "Num2Words", "UnitExpansion"]):
        positives = []
        for op in ops:
            op = eval(op)
            if op not in self.operations: continue
            positives.append(op.generate(text))
        return positives