from .operation import Operation
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from similarity.normalized_levenshtein import NormalizedLevenshtein
import torch


class Pegasus(Operation):
    def __init__(self) -> None:
        super().__init__()
        model_name = 'tuner007/pegasus_paraphrase'
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.DEVICE)
        self.nl = NormalizedLevenshtein()

    def get_response(self, input_text, num_return_sequences=5, beam_groups=5):
        batch = self.tokenizer(input_text, max_length=128, truncation=True, return_tensors="pt").to(self.DEVICE)
        translated = self.model.generate(**batch, max_length=128, 
                            num_beam_groups = beam_groups, diversity_penalty=5.0, 
                            num_beams=beam_groups*2, num_return_sequences=num_return_sequences, temperature=0.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def run_pegasus(self, text, nl=NormalizedLevenshtein(), n=2, soften=False):
        num = n+1 if n>5 else 5
        pegasus_outputs = list(set(self.get_response(text, num_return_sequences=num)))
        pegasus_outputs = [(output, nl.distance(output, text)) for output in pegasus_outputs]
        pegasus_outputs = sorted(pegasus_outputs, key=lambda x: x[1], reverse=True)[:n]
        for pegasus_output, dist in pegasus_outputs:
            if len(pegasus_output.split()) > len(text.split())//2:
                return pegasus_output

    def generate(self, text, **kwargs):
        return self.run_pegasus(text, nl=self.nl)