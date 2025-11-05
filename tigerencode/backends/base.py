"""Base backend for TigerFeat."""

from abc import ABC, abstractmethod
import torch

class TigerFeatBackend(ABC):
    """Abstract base class for TigerFeat backends."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.model = None
        self.transform = None
        self.input_size = None
        self.data_config = None
        self.processor = None

    @abstractmethod
    def initialise(self):
        """Initialize the backend-specific model and transforms."""
        pass

    @abstractmethod
    def prepare_image(self, path):
        """Prepare a single image for forwarding."""
        pass

    @abstractmethod
    def forward(self, input_data):
        """Run forward pass on prepared input."""
        pass

    @abstractmethod
    def get_dataloader_class(self):
        """Return Dataset and Collate classes for batch processing (if supported)."""
        pass

    @abstractmethod
    def info(self):
        """Return backend-specific info."""
        pass