import numpy as np

from ._array_utils import _as_numpy

try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    ig = None
    leidenalg = None
    _import_error = exc
else:
    _import_error = None


def build_edges_from_knn(knn_indices, knn_scores, drop_self=True):
    """
    knn_indices: (N, K)
    knn_scores : (N, K)
    回傳 edge arrays（src/dst/sim）均為 1D array
    """
    nbr_idx = _as_numpy(knn_indices).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(knn_scores).astype(np.float32, copy=False)
    if nbr_idx.shape != nbr_sim.shape:
        raise ValueError("knn_indices shape != knn_scores shape")

    N, K = nbr_idx.shape
    if K == 0:
        return {
            "src": np.zeros((0,), np.int32),
            "dst": np.zeros((0,), np.int32),
            "sim": np.zeros((0,), np.float32),
            "nbr_idx": nbr_idx,
            "nbr_sim": nbr_sim,
        }

    if drop_self:
        keep_mask = nbr_idx != np.arange(N, dtype=np.int32)[:, None]
        rows_all_drop = np.where(~keep_mask.any(axis=1))[0]
        if len(rows_all_drop) > 0:
            keep_mask[rows_all_drop, 0] = True

        src = np.repeat(np.arange(N, dtype=np.int32), K)[keep_mask.ravel()]
        dst = nbr_idx.ravel()[keep_mask.ravel()]
        sim = nbr_sim.ravel()[keep_mask.ravel()]
        return {"src": src, "dst": dst, "sim": sim, "nbr_idx": nbr_idx, "nbr_sim": nbr_sim}

    src = np.repeat(np.arange(N, dtype=np.int32), K)
    dst = nbr_idx.ravel()
    sim = nbr_sim.ravel()
    return {"src": src, "dst": dst, "sim": sim, "nbr_idx": nbr_idx, "nbr_sim": nbr_sim}


def apply_edge_filter_and_weights(
    nbr_idx,
    nbr_sim,
    z_edge=0.5,
    mutual_boost=1.5,
    min_deg=2,
    min_deg_weight=0.8,
    verbose=False,
):
    """
    Leiden 用的建邊策略：
    - global z-score（用所有 sim 的 mean/std）→ mask = (z >= z_edge)
    - mutual kNN 的邊：只做加權（sim * mutual_boost）
    - 最低出度修補：若某點保留邊數 < min_deg，從原始 KNN 的前幾名補回（並把權重打折）
    """
    nbr_idx = _as_numpy(nbr_idx).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(nbr_sim).astype(np.float32, copy=False)
    N, K1 = nbr_idx.shape

    flat = nbr_sim.reshape(-1).astype(np.float32)
    mu = float(flat.mean())
    sigma = float(flat.std()) + 1e-8
    z = (nbr_sim - mu) / sigma
    mask = (z >= float(z_edge))

    if verbose:
        kept = int(mask.sum())
        total = int(mask.size)
        print("[edges] z_edge=%.3f kept %d / %d (%.2f%%)" %
              (z_edge, kept, total, 100.0 * kept / max(1, total)))

    if mutual_boost is None:
        mutual_boost = 1.0
    mutual_boost = float(mutual_boost)

    if mutual_boost != 1.0:
        if verbose:
            print("[edges] computing mutual mask for weighting...")
        nbr_sets = [set(row.tolist()) for row in nbr_idx]
        mutual_mask = np.zeros_like(nbr_sim, dtype=bool)
        for i in range(N):
            cols = nbr_idx[i]
            mutual_mask[i] = np.array([i in nbr_sets[int(j)] for j in cols], dtype=bool)
    else:
        mutual_mask = None

    src = np.repeat(np.arange(N, dtype=np.int32), K1)[mask.ravel()]
    dst = nbr_idx.ravel()[mask.ravel()]
    sim = nbr_sim.ravel()[mask.ravel()]

    if mutual_mask is not None:
        is_mutual = mutual_mask.ravel()[mask.ravel()]
        sim = np.where(is_mutual, sim * mutual_boost, sim)

    if min_deg is not None and int(min_deg) > 0:
        min_deg = int(min_deg)
        deg = np.bincount(src, minlength=N).astype(np.int32)
        need_fix = np.where(deg < min_deg)[0]

        if verbose:
            print("[edges] min_deg=%d nodes needing fix=%d" % (min_deg, int(len(need_fix))))

        if len(need_fix) > 0:
            fix_src = []
            fix_dst = []
            fix_sim = []
            min_deg_weight = float(min_deg_weight)

            existing = {}
            for i in need_fix.tolist():
                existing[i] = set(dst[src == i].tolist())

            for i in need_fix.tolist():
                current_deg = int(deg[i])
                needed = min_deg - current_deg
                if needed <= 0:
                    continue

                max_try = min(K1, needed + 5)
                for k in range(max_try):
                    if current_deg >= min_deg:
                        break
                    j = int(nbr_idx[i, k])
                    if j == i:
                        continue
                    if j in existing[i]:
                        continue
                    score = float(nbr_sim[i, k]) * min_deg_weight
                    fix_src.append(i)
                    fix_dst.append(j)
                    fix_sim.append(score)
                    existing[i].add(j)
                    current_deg += 1

            if len(fix_src) > 0:
                src = np.concatenate([src, np.asarray(fix_src, dtype=np.int32)], axis=0)
                dst = np.concatenate([dst, np.asarray(fix_dst, dtype=np.int32)], axis=0)
                sim = np.concatenate([sim, np.asarray(fix_sim, dtype=np.float32)], axis=0)

                order = np.argsort(-sim)
                src2 = src[order]
                dst2 = dst[order]
                sim2 = sim[order]

                seen = set()
                keep = np.zeros_like(sim2, dtype=bool)
                for t in range(sim2.shape[0]):
                    key = (int(src2[t]), int(dst2[t]))
                    if key in seen:
                        continue
                    seen.add(key)
                    keep[t] = True

                src = src2[keep]
                dst = dst2[keep]
                sim = sim2[keep]

    return {
        "src": src.astype(np.int32, copy=False),
        "dst": dst.astype(np.int32, copy=False),
        "sim": sim.astype(np.float32, copy=False),
    }


def leiden_cluster(src, dst, sim, n_nodes=None, resolution=1.0, seed=0):
    if ig is None or leidenalg is None:
        message = "igraph/leidenalg not available. Please install python-igraph and leidenalg."
        if _import_error is not None:
            raise ImportError(message) from _import_error
        raise ImportError(message)

    src = _as_numpy(src).astype(np.int32, copy=False)
    dst = _as_numpy(dst).astype(np.int32, copy=False)
    sim = _as_numpy(sim).astype(float, copy=False)

    if n_nodes is None:
        if src.size == 0 and dst.size == 0:
            n_nodes = 0
        else:
            n_nodes = int(max(int(src.max(initial=0)), int(dst.max(initial=0))) + 1)

    g = ig.Graph(n=int(n_nodes))
    if src.size > 0:
        g.add_edges(list(zip(src.tolist(), dst.tolist())))
        g.es["weight"] = sim.tolist()
    else:
        g.es["weight"] = []

    try:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
    except TypeError:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=float(resolution),
        )

    cluster_id = np.zeros(int(n_nodes), dtype=np.int32)
    for cid, nodes in enumerate(partition):
        for v in nodes:
            cluster_id[int(v)] = int(cid)

    return cluster_id, g
