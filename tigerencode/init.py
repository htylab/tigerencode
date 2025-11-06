"""Helper utilities for constructing TigerEncode models."""

from .config import TigerEncodeConfig, _normalise_model_kwargs
from .model import TigerEncodeImageModel, TigerEncodeTextModel


def model_img(**kwargs):
    """Initialise an image encoding model."""

    normalised = _normalise_model_kwargs(kwargs)
    resolved = dict(normalised)
    resolved.setdefault("model", "timm@resnet50")
    config = TigerEncodeConfig(**resolved)
    return TigerEncodeImageModel(config)


def model_text(**kwargs):
    """Initialise a text encoding model."""

    normalised = _normalise_model_kwargs(kwargs)
    resolved = dict(normalised)
    resolved.setdefault("model", "hf@Linq-AI-Research/Linq-Embed-Mistral")
    config = TigerEncodeConfig(**resolved)
    return TigerEncodeTextModel(config)


def init(**kwargs):
    """Backward compatible helper that mirrors :func:`model_img`."""

    return model_img(**kwargs)
