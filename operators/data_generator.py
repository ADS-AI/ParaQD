from .positive_ops import PositiveSamples
from .negative_ops import NegativeSamples


class DataGenerator(object):
    def __init__(self):
        self.positive_generator = PositiveSamples()
        self.negative_generator = NegativeSamples()

    def generate(self, text, positive_ops=["BackTranslate", "SameSentence", "Num2Words", "UnitExpansion"], 
                negative_ops=["MostImportantPhraseRemover", "DeleteLastSentence", "ReplaceNamedEntities",
                            "ReplaceNumericalEntities", "Pegasus", "ReplaceUnits", "NegateQuestion"]):
        positives = self.positive_generator.generate(text, ops=positive_ops)
        negatives = self.negative_generator.generate(text, ops=negative_ops)
        return positives, negatives
