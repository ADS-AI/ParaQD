from .operation import Operation
import numpy as np
import pickle
import random


class TF_IDF_Replacement(Operation):
    def __init__(self, tfidf_path) -> None:
        super().__init__()
        self.tfidf = pickle.load(open(tfidf_path, "rb"))
        self.words = self.tfidf.get_feature_names()

    def __sample(self, n=3):
        return random.sample(self.words, n)

    def generate(self, text, soften=False, **kwargs):
        transformed = self.tfidf.transform([text]).toarray()
        most_imp = np.argpartition(transformed, -4)[:, -4:]
        array = most_imp[0]
        question = text
        vals = []
        num_replace = np.random.randint(1, 3)
        if soften:
            num_replace = 1
        replacements = self.__sample(num_replace)
        for idx in array:
            val = transformed[0][idx]
            word = self.words[idx]
            vals.append((val, word))
        vals.sort(reverse = True)
        replaced = list(map(lambda x: x[1], vals))[:num_replace]
        for replaced_, replacement in zip(replaced, replacements):
            question = question.replace(replaced_, replacement, 1)
        return question