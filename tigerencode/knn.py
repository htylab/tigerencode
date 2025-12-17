import numpy as np

from ._array_utils import _as_numpy, _l2_normalize_rows

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


def compute_knn_similarity(
    embeddings,
    topk=150,
    chunk_size=20000,
    ensure_normalized=True,
    use_torch=True,
    device="cuda",
    verbose=False,
):
    """
    回傳：
      indices: (N, topk) int32
      scores : (N, topk) float32
    備註：topk 通常會包含自己（第一名），後續可選擇丟掉 self-loop。
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))

    if ensure_normalized:
        X = _l2_normalize_rows(X).astype(np.float32, copy=False)

    N = X.shape[0]
    if topk >= N:
        topk = max(1, N - 1)

    if use_torch and torch is not None:
        ok = True
        try:
            if device == "cuda" and not torch.cuda.is_available():
                ok = False
        except Exception:
            ok = False

        if ok:
            if verbose:
                print("[KNN] using torch (%s)" % device)
            X_t = torch.from_numpy(X)
            if device:
                X_t = X_t.to(device)

            all_vals = np.empty((N, topk), dtype=np.float32)
            all_idxs = np.empty((N, topk), dtype=np.int32)

            for start in range(0, N, chunk_size):
                end = min(N, start + chunk_size)
                chunk = X_t[start:end]  # (B, D)
                sims = chunk @ X_t.T  # (B, N)
                vals, idxs = torch.topk(sims, k=topk, dim=1)

                all_vals[start:end] = vals.detach().float().cpu().numpy()
                all_idxs[start:end] = idxs.detach().cpu().numpy().astype(np.int32)

            return {"indices": all_idxs, "scores": all_vals}

    if verbose:
        print("[KNN] torch/cuda not available, using CPU numpy (may be slow)")

    all_vals = np.empty((N, topk), dtype=np.float32)
    all_idxs = np.empty((N, topk), dtype=np.int32)

    Xt = X.T
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        chunk = X[start:end]  # (B, D)
        sims = np.matmul(chunk, Xt)  # (B, N)

        part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]  # (B, topk)
        part_sims = np.take_along_axis(sims, part, axis=1)  # (B, topk)
        order = np.argsort(-part_sims, axis=1)
        idxs = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(part_sims, order, axis=1)

        all_idxs[start:end] = idxs.astype(np.int32)
        all_vals[start:end] = vals.astype(np.float32)

    return {"indices": all_idxs, "scores": all_vals}


def drop_self_from_knn(knn_indices, knn_scores):
    """
    輸入：KNN (N, K)（通常第一名是自己）
    輸出：去掉第一欄（若 K>=2），否則原樣回傳
    """
    idx = _as_numpy(knn_indices).astype(np.int32, copy=False)
    sim = _as_numpy(knn_scores).astype(np.float32, copy=False)
    if idx.ndim != 2 or sim.ndim != 2 or idx.shape != sim.shape:
        raise ValueError("knn_indices/knn_scores must be (N,K) with same shape")

    if idx.shape[1] >= 2:
        return idx[:, 1:], sim[:, 1:]
    return idx, sim
