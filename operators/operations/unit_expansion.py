import inflect
from similarity.normalized_levenshtein import NormalizedLevenshtein
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from .operation import Operation


class UnitChange(Operation):
    def __init__(self):
        currencies = ["dollar", "cent", "nickel", "penny", "quarter", "dime", "rupee", "paisa", "pound", "euro"]
        currency_abbrv = {}
        currency_symbs = ["$", "£", "₹", "€"]
        currency_symbs_abbrv = {}
        currency_short = ["rs", "USD", "EUR", "GBP"]
        currency_short_abbrv = {}
        length = ["inch", "meter", "metre", "kilometre", "kilometer", "centimeter", "millimeter", "millimetre", "centimetre", "foot", "mile"]
        length_abbrv = {"m": "metre", "km": "kilometre", "cm":"centimetre", "mm":"millimetre", "ft": "foot", "mil": "mile"}
        weight = ["gram", "kilogram"]
        weight_abbrv = {"g": "gram", "gm": "gram", "kg": "kilogram"}
        time = ["minute", "hour", "day", "year", "month", "week"]
        time_abbrv = {"min": "minute", "hr": "hour", "yr": "year", "wk": "week"}
        speed = ["kilometre per hour", "miles per hour", "metre per second"]
        speed_abbrv = {"km/hr": "kilometre per hour", "km/h": "kilometre per hour", "kmph": "kilometre per hour", "mph": "miles per hour", "m/s": "metre per second", "mps": "metre per second"}
        p = inflect.engine()
        self.full = [currencies, currency_symbs, currency_short, length, weight, time, speed]
        self.plurals = [[p.plural(unit) for unit in unit_type] for unit_type in self.full]
        self.abbreviated = [currency_abbrv, currency_symbs_abbrv, currency_short_abbrv, length_abbrv, weight_abbrv, time_abbrv, speed_abbrv]
        self.abbreviated_pl = [{p.plural(key): p.plural(value) for (key, value) in ab_dict.items()} for ab_dict in self.abbreviated]
        self.nl = NormalizedLevenshtein()

    def find_closest(self, word, wordlist):
        if word in wordlist:
            return word
        closest = min([(w, self.nl.distance(word, w)) for w in wordlist], key=lambda x: x[1])
        return closest[0]

    def untokenize(self, words):
        """
        ref: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py#L28
        """
        text = " ".join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()

    def abbreviate(self, word, abbrvs):
        closest_word = self.find_closest(word, abbrvs.values())
        for a, w in abbrvs.items():
            if closest_word == w:
                return a
        return word

    def find_replacement_full_index(self, unit, index, negative=True):
        unit_list = self.full[index]
        unit_list_pl = self.plurals[index]
        unit = unit.lower().strip()
        
        if unit in unit_list:
            replacement = np.random.choice(unit_list)
            if negative:
                while replacement == unit or self.nl.distance(unit, replacement)<0.2:
                    replacement = np.random.choice(unit_list)
            else:
                if unit == "rs": return "Rupees"
                replacement = unit
            return replacement
            
        if unit in unit_list_pl:
            replacement = np.random.choice(unit_list_pl)
            if negative:
                while replacement == unit or self.nl.distance(unit, replacement)<0.2:
                    replacement = np.random.choice(unit_list)
            else:
                replacement = unit
            return replacement
        return ""

    def find_replacement_full(self, unit, negative=True):
        for index in range(len(self.full)):
            replacement = self.find_replacement_full_index(unit, index, negative)
            if replacement:
                return replacement
        return ""

    def find_replacement_abbreviated(self, unit, negative=True):
        unit = unit.strip().lower()

        for i, abb_type in enumerate(self.abbreviated):
            abb_type_pl = self.abbreviated_pl[i]
            if unit in abb_type:
                unit_full = abb_type[unit]
                replacement = self.find_replacement_full_index(unit_full, i, negative=negative)
                if negative:
                    replacement = self.abbreviate(replacement, abb_type)
                return replacement

            if unit in abb_type_pl:
                unit_full = abb_type_pl[unit]
                replacement = self.find_replacement_full_index(unit_full, i, negative=negative)
                if negative:
                    replacement = self.abbreviate(replacement, abb_type_pl)
                return replacement

        return ""

    def find_replacement(self, unit, negative=True, abbreviated=False):
        if abbreviated:
            return self.find_replacement_abbreviated(unit, negative)
        replacement = self.find_replacement_full(unit, negative)
        if replacement:
            return replacement
        return self.find_replacement_abbreviated(unit, negative)

    def change(self, sent, negative=True):
        original_sent = sent
        tokenized = word_tokenize(sent)
        replaced = False

        for i, token in enumerate(tokenized):
            quants = re.findall("\d+([a-z/]{1,5})", token)
            if quants:
                if replaced and np.random.choice([True, False], p=[0.65, 0.35]): break
                unit = quants[0]
                replacement = self.find_replacement(unit, abbreviated=True, negative=negative)
                if replacement == "":
                    continue
                token_new = token.replace(unit, replacement)
                tokenized[i] = token_new
                replaced = True    
            else:
                replacement = self.find_replacement(token, negative=negative)
                if replacement:
                    if replaced and np.random.choice([True, False], p=[0.65, 0.35]): break
                    tokenized[i] = replacement
                    replaced = True
                else:
                    continue
        
        sent = self.untokenize(tokenized)
        if sent == original_sent:
            sent = ""
        return sent

    def generate(self, text, **kwargs):
        return self.change(text, negative=False)