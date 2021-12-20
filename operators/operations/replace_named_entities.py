from .operation import Operation
import spacy
import random
import numpy as np
import re
import os


class ReplaceNamedEntities(Operation):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        resource_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
        fpaths = [os.path.join(resource_dir, fpath) for fpath in os.listdir(resource_dir) if fpath.endswith(".txt")]
        self.resources = [open(fpath, "r").read().splitlines() for fpath in fpaths]

    def get_replacement(self, entity):
        possibilities = []
        for resource in self.resources:
            if entity.lower() in list(map(lambda x: x.lower(), resource)):
                possibilities.extend(random.sample(resource, 5))
                break
        possibilities = [" {} ".format(x) for x in possibilities]
        possibilities.append(" ")
        return random.choice(possibilities)

    def replace_named_entities(self, text, soften=False):
        """
        loses persons, organizations, products and places
        """
        doc = self.nlp(text)
        named_entities = set(["PERSON", "ORG", "PRODUCT", "EVENT", "GPE", "GEO"])
        ne = []
        for x in doc.ents:
            if x.label_ in named_entities:
                ne.append((x.text, x.start_char, x.end_char))
        if len(ne) == 0:
            return text
        ne_new = random.sample(ne, np.random.randint(1, min(len(ne), 3)+1))
        ne_new = sorted(ne_new, key=lambda x: x[1])
        if soften:
            ne_new = ne_new[:1]
        shift = 0
        for (entity, start, end) in ne_new:
            replacemnt = self.get_replacement(entity)
            text = text[:start-shift] + replacemnt + text[end-shift:]
            shift += end - start - len(replacemnt)
        text = re.sub("\s+", " ", text)
        return text.strip()

    def generate(self, text, **kwargs):
        soften = kwargs.get("soften", False)
        return self.replace_named_entities(text, soften)