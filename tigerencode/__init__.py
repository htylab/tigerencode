"""TigerEncode public API."""

from .adaptor import ProjectionAdaptor
from .cluster import embed_clustering_leiden, knn_merge, strict_dedup
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
    "embed_clustering_leiden",
    "knn_merge",
    "strict_dedup",
]

__version__ = "0.1.0"
