"""TigerEncode public API."""

from .adaptor import ProjectionAdaptor
from .init import init, model_img, model_text
from .model import TigerEncodeImageModel, TigerEncodeTextModel

__all__ = [
    "__version__",
    "init",
    "model_img",
    "model_text",
    "ProjectionAdaptor",
    "TigerEncodeImageModel",
    "TigerEncodeTextModel",
]

__version__ = "0.1.0"
