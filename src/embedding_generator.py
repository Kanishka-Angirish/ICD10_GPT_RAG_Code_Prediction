from transformers import BioGptTokenizer
import torch
import numpy as np


class EmbeddingGenerator:

    def __init__(self, embed_model):
        self.model = embed_model
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt-large", clean_up_tokenization_spaces=True)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def generate_embeddings(self, text):
        encoded_input = self.tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors='pt')
        encoded_input = {key: val.to("mps") for key, val in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input, output_hidden_states=True)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy()
