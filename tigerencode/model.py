"""Core TigerFeat model."""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .config import TigerFeatConfig, _normalise_model_kwargs
from .backends.base import TigerFeatBackend

# Lazy import backends
def _get_backend_class(backend_name):
    if backend_name == "timm":
        from .backends.timm import TimmBackend
        return TimmBackend
    elif backend_name == "xray":
        from .backends.xray import XrayBackend
        return XrayBackend
    elif backend_name == "hf":
        from .backends.hf import HfBackend
        return HfBackend
    else:
        raise ValueError(f"Unsupported backend: {backend_name}")

class TigerFeatModel(object):
    """Lightweight wrapper for vision models with feat extraction."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = TigerFeatConfig(**_normalise_model_kwargs(kwargs))
        elif kwargs:
            raise ValueError("Provide either a config object or keyword arguments, not both.")

        self.config = config
        self.device = self._resolve_device(config.device)

        if "@" not in self.config.model:
            raise ValueError("Model must be in '<backend>@<model_name>' format.")
        backend_name, model_name = self.config.model.split("@", 1)
        self.backend_name = backend_name.strip().lower()
        self.model_name = model_name.strip()
        if not self.backend_name or not self.model_name:
            raise ValueError("Model must include backend and name separated by '@'.")

        BackendClass = _get_backend_class(self.backend_name)
        self.backend = BackendClass(self.config, self.device)
        self.backend.initialise()

    @staticmethod
    def _resolve_device(device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _safe_open_pil(self, path):
        if not isinstance(path, str):
            raise TypeError("Image input must be a file path string.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to open image {path}: {e}")

    def feat(self, image_path):
        if not isinstance(image_path, str):
            raise TypeError("Image input must be a file path string.")
        try:
            input_data = self.backend.prepare_image(image_path)
            feats = self.backend.forward(input_data)
            return feats.detach().cpu().squeeze(0).numpy()
        except Exception as e:
            print(f"[Warning] Skipped {image_path}: {e}")
            return np.empty((0,), dtype=np.float32)

    def feat_batch(
        self,
        images,
        batch_size=16,
        num_workers=None,
        pin_memory=True,
        show_progress=None,
        use_dataloader=None,
        dataloader_threshold=1000,
    ):
        if isinstance(images, str):
            images = [images]
        elif not isinstance(images, (list, tuple)):
            raise TypeError("Input must be a list/tuple of file paths.")

        images = [p for p in images if isinstance(p, str)]
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32), []

        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            _tqdm = None
        if show_progress is None:
            show_progress = n >= dataloader_threshold

        if use_dataloader is None:
            use_dataloader = n >= dataloader_threshold

        if num_workers is None:
            num_workers = 8 if self.device.type == "cuda" else 0

        all_feats = []
        filelist = []

        DatasetClass, CollateClass = self.backend.get_dataloader_class()

        if DatasetClass is not None and use_dataloader:
            dataset = DatasetClass(images, self.backend.transform)
            collate = CollateClass()
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory if self.device.type == "cuda" else False,
                collate_fn=collate,
            )
            iterator = _tqdm(loader, desc="Extracting") if (show_progress and _tqdm) else loader
            with torch.no_grad():
                for batch, paths in iterator:
                    if batch.numel() == 0:
                        continue
                    batch = batch.to(self.device)
                    feats = self.backend.forward(batch)
                    all_feats.append(feats.detach().cpu())
                    filelist.extend(paths)
        else:
            # Fallback to loop (e.g., for HF)
            rng = _tqdm(range(0, n, batch_size), desc="Extracting") if (show_progress and _tqdm) else range(0, n, batch_size)
            for start in rng:
                chunk = images[start:start + batch_size]
                inputs_list, valids = [], []
                for p in chunk:
                    try:
                        if self.backend_name == "hf":
                            inputs_list.append(self._safe_open_pil(p))
                        else:
                            inputs_list.append(self.backend.prepare_image(p))
                        valids.append(p)
                    except Exception as e:
                        print(f"[Warning] Skipped {p}: {e}")
                if not inputs_list:
                    continue
                if self.backend_name == "hf":
                    inputs = self.backend.processor(images=inputs_list, return_tensors="pt").to(self.device)
                    feats = self.backend.forward(inputs)
                else:
                    batch = torch.cat(inputs_list, dim=0)
                    feats = self.backend.forward(batch)
                all_feats.append(feats.detach().cpu())
                filelist.extend(valids)

        if not all_feats:
            return np.empty((0,), dtype=np.float32), []

        feats = torch.cat(all_feats, dim=0).numpy()
        return feats, filelist

    def info(self):
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "device": str(self.device),
            **self.backend.info(),
        }