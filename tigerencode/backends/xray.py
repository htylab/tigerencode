"""X-ray backend for TigerFeat."""

import numpy as np
import torch
import torch.nn as nn
import torchxrayvision as xrv
from skimage import io as skio
import torchvision.transforms as tv_transforms
from torch.utils.data import Dataset
from .base import TigerFeatBackend

class XrayBackend(TigerFeatBackend):
    def initialise(self):
        self.model = xrv.models.get_model(self.config.model.split("@", 1)[1])
        self.model.eval()
        self.model.to(self.device)

        size = getattr(self.model, "img_size", 224)
        if isinstance(size, (tuple, list)):
            size = size[0]
        size = int(size)
        self.input_size = (1, size, size)
        self.transform = tv_transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(size),
            ]
        )

    def prepare_image(self, path):
        img = skio.imread(path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 2:
            img = img[None, ...]
        else:
            img = img.mean(2)[None, ...]
        img = img.astype("float32", copy=False)
        transformed = self.transform(img)
        if isinstance(transformed, torch.Tensor):
            tensor = transformed.float().unsqueeze(0)
        else:
            tensor = torch.from_numpy(np.asarray(transformed, dtype="float32")).unsqueeze(0)
        return tensor.to(self.device)

    def forward(self, tensor):
        with torch.no_grad():
            feat = self.model.features(tensor)
            feat = nn.AdaptiveAvgPool2d((1, 1))(feat)
            feat = feat.view(feat.size(0), -1)
        return feat

    def get_dataloader_class(self):
        class XrayDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                path = self.paths[idx]
                try:
                    img = skio.imread(path)
                    img = xrv.datasets.normalize(img, 255)
                    if img.ndim == 2:
                        img = img[None, ...]
                    else:
                        img = img.mean(2)[None, ...]
                    img = img.astype("float32", copy=False)
                    out = self.transform(img)
                    t = torch.from_numpy(np.asarray(out, dtype="float32"))
                    return t, path
                except Exception as e:
                    print(f"[Warning] Skipped {path}: {e}")
                    return None

        class Collate:
            def __call__(self, batch):
                batch = [b for b in batch if b is not None]
                if not batch:
                    return torch.empty(0), []
                tensors, paths = zip(*batch)
                return torch.stack(tensors), list(paths)

        return XrayDataset, Collate

    def info(self):
        return {
            "input_size": self.input_size,
        }