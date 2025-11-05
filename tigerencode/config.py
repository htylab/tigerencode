"""Configuration for TigerFeat."""

class TigerFeatConfig(object):
    """Simple configuration container for TigerFeatModel."""

    def __init__(
        self,
        model="timm@resnet50",
        pretrained=True,
        device=None,
        transform_kwargs=None,
    ):
        self.model = model
        self.pretrained = pretrained
        self.device = device
        self.transform_kwargs = transform_kwargs or {}


def _normalise_model_kwargs(kwargs):
    if "model_name" in kwargs and "model" not in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["model"] = new_kwargs.pop("model_name")
        return new_kwargs
    return kwargs