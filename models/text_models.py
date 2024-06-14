import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class BERT(nn.Module):
    """
    BERT-like model for text embeddings

    ========================
    Initialization Parameters:
    text_embedding_model (str): name of the BERT-like model to use

    ========================
    forward Inputs:
    tokens (dict): dictionary of tokenized text

    ========================
    forward Outputs:
    output_dict (dict): dictionary of embeddings, including:
        - clip_emb (torch.tensor): B x D tensor of CLS token embeddings
        - time_emb (torch.tensor): B x T x D tensor of token embeddings
        - mean_emb (torch.tensor): B x D tensor of mean-pooled token embeddings (DEFAULT used)
        - time_mask (torch.tensor): B x T tensor of attention masks
    """
    def __init__(self, text_embedding_model: str = 'roberta-base'):
        super().__init__()
        self.model = AutoModel.from_pretrained(text_embedding_model)
        self.embed_dim = self.model.config.hidden_size

    def mean_pooling(self, model_output, attention_mask):
        # adapted from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, **tokens):
        output = self.model(**tokens)
        mean_emb = self.mean_pooling(output, tokens["attention_mask"])
        mean_emb = F.normalize(mean_emb, p=2, dim=1)
        # [CLS] pooling
        clip_emb = output.last_hidden_state[:, 0, :]
        # clip_emb = F.normalize(clip_emb, p=2, dim=1)
        time_emb = output.last_hidden_state
        output_dict = {
            "clip_emb": clip_emb,
            "time_emb": time_emb,
            "mean_emb": mean_emb,
            "time_mask": tokens["attention_mask"]
        }
        return output_dict