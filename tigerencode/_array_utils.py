import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _l2_normalize_rows(x, eps=1e-12):
    x = _as_numpy(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)
