"""Base backend for TigerEncode text models."""

from abc import ABC, abstractmethod


class TigerEncodeTextBackend(ABC):
    """Abstract base class for TigerEncode text backends."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def initialise(self):
        """Initialise the backend resources such as model and tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def prepare_text(self, text):
        """Prepare a single text string for forwarding."""
        raise NotImplementedError

    def prepare_text_batch(self, texts):
        """Prepare a batch of text strings for forwarding."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs):
        """Run a forward pass for the provided inputs."""
        raise NotImplementedError

    def info(self):
        """Return backend specific information."""
        return {}
