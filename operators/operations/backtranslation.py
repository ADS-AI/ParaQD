import torch
from .operation import Operation
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from sentence_splitter import SentenceSplitter
from nltk.tokenize import word_tokenize, sent_tokenize
from similarity.normalized_levenshtein import NormalizedLevenshtein
import os


class BackTranslate(Operation):
    def __init__(self):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("self.DEVICE", self.DEVICE)
        forward_tokenizer_path = "/content/drive/MyDrive/FSMT/en-de/tokenizer"
        forward_model_path = "/content/drive/MyDrive/FSMT/en-de/model"
        if not os.path.exists(forward_tokenizer_path):
            forward_tokenizer_path = forward_model_path = "facebook/wmt19-en-de"
        backward_tokenizer_path = "/content/drive/MyDrive/FSMT/de-en/tokenizer"
        backward_model_path = "/content/drive/MyDrive/FSMT/de-en/model"
        if not os.path.exists(backward_tokenizer_path):
            backward_tokenizer_path = backward_model_path = "facebook/wmt19-de-en"
        print("[INFO] Loading en-de model...")
        self.forward_tokenizer = FSMTTokenizer.from_pretrained(forward_tokenizer_path)
        self.forward_model = FSMTForConditionalGeneration.from_pretrained(forward_model_path).to(self.DEVICE)
        print("[INFO] Loading de-en model...")
        self.backward_tokenizer = FSMTTokenizer.from_pretrained(backward_tokenizer_path)
        self.backward_model = FSMTForConditionalGeneration.from_pretrained(backward_model_path).to(self.DEVICE)
        self.splitter = SentenceSplitter(language='en')

    def beam_search(self, text, tokenizer, model, num_beams=10):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(self.DEVICE)
        beam_output = model.generate(
            input_ids, 
            num_beams=num_beams, 
            early_stopping=True)
        text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
        return text

    def diverse_beam_search(self, text, tokenizer, model, beam_groups=4, num_return_sequences=4, diversity_penalty=1.5):
        input_ids = tokenizer.encode(text, return_tensors="pt").to(self.DEVICE)
        beam_outputs = model.generate(
            input_ids,
            num_beams = beam_groups*3,
            num_return_sequences = num_return_sequences,
            diversity_penalty = diversity_penalty,
            num_beam_groups = beam_groups)
        text = []
        for beam_output in beam_outputs:
            text.append(tokenizer.decode(beam_output, skip_special_tokens=True))
        return text

    def backtranslate_single(self, text):
        nl = NormalizedLevenshtein()
        t_split = sent_tokenize(text)
        paraphrased = [""]
        for sentence in t_split:
            temp = []
            translated = self.beam_search(sentence, self.forward_tokenizer, self.forward_model, num_beams=8)
            paraphrases = self.diverse_beam_search(translated, self.backward_tokenizer, self.backward_model, num_return_sequences=2, beam_groups=4, diversity_penalty=50.0)
            for paraphrase in paraphrases:
                for prev in paraphrased:
                    temp.append(prev+paraphrase)
            paraphrased = temp
        paraphrased = [(paraphrase, nl.distance(text, paraphrase)) for paraphrase in paraphrased]
        paraphrase = sorted(paraphrased, key=lambda x: x[1], reverse=True)[0]
        return paraphrase[0]

    def generate(self, texts, **kwargs):
        paraphrases = []
        for text in texts:
            paraphrase = self.backtranslate_single(text)
            paraphrases.append(paraphrase)
        return paraphrases