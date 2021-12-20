from .operation import Operation
from sentence_transformers import SentenceTransformer, util
import re
import pke
import numpy as np



class MostImportantPhraseRemover(Operation):
    def __init__(self) -> None:
        super().__init__()
        self.sentence_transformer = SentenceTransformer("paraphrase-mpnet-base-v2")

    def lose_most_important_word(self, text, soften=False):
        tokenized = text.split()
        original_embedding = self.sentence_transformer.encode([text], convert_to_tensor=True)
        min_score = 1; final_text = ""
        processed_sentences = []
        keyphrases = []

        try:
            self.extractor = pke.unsupervised.TopicRank()
            self.extractor.load_document(input=text, language='en')
            self.extractor.candidate_selection()
            self.extractor.candidate_weighting()
            keyphrases = self.extractor.get_n_best(n=4)
            keyphrases = list(map(lambda x: x[0], keyphrases))
        except:
            pass

        if len(keyphrases) > 0:
            for keyphrase in keyphrases:
                processed_text = text.replace(keyphrase, " ")
                processed_text = re.sub("\s+", " ", processed_text)
                processed_sentences.append(processed_text)
        
        else:
            for idx, word in enumerate(tokenized):
                processed_text = " ".join(tokenized[:idx]+tokenized[idx+1:])
                processed_text = re.sub("\s+", " ", processed_text)
                processed_sentences.append(processed_text)

        processed_embeddings = self.sentence_transformer.encode(processed_sentences, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(original_embedding, processed_embeddings)
        idx = np.argmin(cosine_scores[0].cpu().numpy())
        return processed_sentences[idx]


    def generate(self, text, **kwargs):
        return self.lose_most_important_word(text)