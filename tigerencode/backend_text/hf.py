"""Hugging Face backend for TigerEncode text models."""

import torch
from transformers import AutoModel, AutoTokenizer

from .base import TigerEncodeTextBackend


class HfTextBackend(TigerEncodeTextBackend):
    """Backend that uses Hugging Face transformer models for text embeddings."""

    def initialise(self):
        model_id = self.config.model.split("@", 1)[1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def prepare_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        return inputs.to(self.device)

    def prepare_text_batch(self, texts):
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return inputs.to(self.device)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Unsupported output structure for Hugging Face text model.")
        return feat
