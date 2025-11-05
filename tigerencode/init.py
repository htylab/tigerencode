from .config import TigerFeatConfig
from .model import TigerFeatModel

def init(**kwargs):
    config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
    return TigerFeatModel(config)