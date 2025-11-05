"""Hugging Face backend for TigerFeat."""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from .base import TigerFeatBackend

class HfBackend(TigerFeatBackend):
    def initialise(self):
        self.processor = AutoProcessor.from_pretrained(self.config.model.split("@", 1)[1])
        self.model = AutoModel.from_pretrained(self.config.model.split("@", 1)[1]).to(self.device)
        self.model.eval()

        processor = getattr(self.processor, "image_processor", self.processor)
        size = getattr(processor, "size", None)
        channels = getattr(processor, "num_channels", None)
        if channels is None:
            image_mean = getattr(processor, "image_mean", None)
            if isinstance(image_mean, (list, tuple)) and image_mean:
                channels = len(image_mean)
        if channels is None:
            channels = 3

        resolved_size = None
        if isinstance(size, dict):
            height = size.get("height")
            width = size.get("width")
            shortest = (
                size.get("shortest_edge")
                or size.get("shortest_side")
                or size.get("shortest")
            )
            if height is not None and width is not None:
                resolved_size = (channels, int(height), int(width))
            elif shortest is not None:
                resolved_size = (channels, int(shortest), int(shortest))
        elif isinstance(size, (tuple, list)) and len(size) >= 2:
            resolved_size = (channels, int(size[-2]), int(size[-1]))
        elif isinstance(size, int):
            resolved_size = (channels, int(size), int(size))

        self.input_size = resolved_size or (channels, 224, 224)

    def prepare_image(self, path):
        with Image.open(path) as img:
            img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.to(self.device)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output"):
                feat = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                feat = outputs.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Unknown output structure for HF model.")
        return feat

    def get_dataloader_class(self):
        # HF doesn't use DataLoader in the same way; return None to fall back to loop
        return None, None

    def info(self):
        return {
            "input_size": self.input_size,
        }