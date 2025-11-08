"""Helper utilities for constructing TigerEncode models."""

from .config import TigerEncodeConfig
from .model import TigerEncodeImageModel, TigerEncodeTextModel


def model_img(**kwargs):
    """Initialise an image encoding model."""

    adaptor = kwargs.pop("adaptor", None)
    resolved = dict(kwargs)
    resolved.setdefault("model", "timm@resnet50")
    config = TigerEncodeConfig(**resolved)
    return TigerEncodeImageModel(config, adaptor=adaptor)


def model_text(**kwargs):
    """Initialise a text encoding model."""

    resolved = dict(kwargs)
    resolved.setdefault("model", "hf@Linq-AI-Research/Linq-Embed-Mistral")
    config = TigerEncodeConfig(**resolved)
    return TigerEncodeTextModel(config)


def init(**kwargs):
    """Backward compatible helper that mirrors :func:`model_img`."""

    return model_img(**kwargs)
