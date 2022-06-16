from .combined_operations import CombinedOperation
from .operations import MostImportantPhraseRemover, DeleteLastSentence, ReplaceNamedEntities, \
                        ReplaceNumericalEntities, Pegasus, ReplaceUnits, NegateQuestion, TF_IDF_Replacement


class NegativeSamples(CombinedOperation):
    def __init__(self):
        super().__init__()
        self.operations = {
            "MostImportantPhraseRemover": MostImportantPhraseRemover(),
            "DeleteLastSentence": DeleteLastSentence(),
            "ReplaceNamedEntities": ReplaceNamedEntities(),
            "ReplaceNumericalEntities": ReplaceNumericalEntities(),
            "Pegasus": Pegasus(),
            "ReplaceUnits": ReplaceUnits(),
        }
        self.backup = TF_IDF_Replacement()

    def operate(self, text, operator, soften=False, c=0):
        initial_text = text
        initial_wordlen = len(text.split())
        text = operator.generate(text, soften=soften)
  
        if text is None or len(text)<2 or text == initial_text:
            text = self.backup.generate(initial_text, soften=soften)
            return text

        if text and len(text.split()) < initial_wordlen//2 and c < 5:
            c += 1
            text = self.lose_information_single(initial_text, operator, soften=True, c=c)

        return text

    def generate(self, text, ops=["MostImportantPhraseRemover", "DeleteLastSentence", "ReplaceNamedEntities",
                                  "ReplaceNumericalEntities", "Pegasus", "ReplaceUnits"], 
                                  **kwargs):
        negatives = []
        for op in ops:
            if op not in self.operations: continue
            operator = self.operations[op]
            negatives.append(self.operate(text, operator))
        return negatives 