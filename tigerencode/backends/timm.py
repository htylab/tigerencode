"""TIMM backend for TigerFeat."""

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from torch.utils.data import Dataset
from .base import TigerFeatBackend

class TimmBackend(TigerFeatBackend):
    def initialise(self):
        self.model = timm.create_model(
            self.config.model.split("@", 1)[1],
            pretrained=self.config.pretrained,
        )
        self.model.eval()
        self.model.to(self.device)

        self.data_config = resolve_data_config({}, model=self.model)
        transform_kwargs = dict(self.config.transform_kwargs)
        transform_kwargs.setdefault("is_training", False)
        self.transform = create_transform(**self.data_config, **transform_kwargs)
        input_size = self.data_config.get("input_size")
        self.input_size = tuple(input_size) if input_size is not None else None

    def prepare_image(self, path):
        with Image.open(path) as img:
            img = img.convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def forward(self, tensor):
        with torch.no_grad():
            feats = self.model.forward_features(tensor)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            elif isinstance(feats, dict):
                feats = self._select_features_from_dict(feats)
            if hasattr(self.model, "forward_head"):
                try:
                    feats = self.model.forward_head(feats, pre_logits=True)
                except TypeError:
                    feats = self.model.forward_head(feats)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            elif isinstance(feats, dict):
                feats = self._select_features_from_dict(feats)
            if isinstance(feats, torch.Tensor) and feats.ndim == 4:
                feats = feats.mean(dim=(2, 3))
        return feats

    def get_dataloader_class(self):
        class TimmDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                path = self.paths[idx]
                try:
                    with Image.open(path) as f:
                        img = f.convert("RGB")
                    return self.transform(img), path
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

        return TimmDataset, Collate

    def info(self):
        return {
            "input_size": self.input_size,
        }

    @staticmethod
    def _select_features_from_dict(features_dict):
        if not features_dict:
            raise ValueError("Feature dictionary is empty and cannot be reduced.")
        preferred_keys = [
            "x_norm_clstoken",
            "x_norm_cls_token",
            "cls_token",
            "class_token",
            "image_features",
            "pooled_features",
            "features",
        ]
        for key in preferred_keys:
            if key in features_dict:
                return features_dict[key]
        return next(reversed(features_dict.values()))