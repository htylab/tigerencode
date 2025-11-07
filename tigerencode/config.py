"""Configuration for TigerEncode models."""


class TigerEncodeConfig(object):
    """Simple configuration container for TigerEncode models."""

    def __init__(
        self,
        model="",
        pretrained=True,
        device=None,
        transform_kwargs=None,
        token=None,
    ):
        self.model = model
        self.pretrained = pretrained
        self.device = device
        self.transform_kwargs = transform_kwargs or {}
        self.token = token
