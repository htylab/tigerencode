#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding clustering and deduplication utilities.
embed_clustering.py（legacy 風格，無 typing / 無 dataclass）

包含兩條路徑：

A) Leiden clustering（群集）
   embeddings -> KNN -> edge 篩選/加權 -> Leiden -> cluster_id

B) mutual-k merge（近重複壓縮 / 去重）
   - 核心：mutualk_merge_from_knn_sim(nbr_idx, nbr_sim, ...)
   - 單輪 step：merge_step_from_embeddings(...) 或 merge_step_from_knn(...)
   - 多輪：knn_merge(...)（反覆 step，維護 orig_to_cur / cap / history / rep_index）

公開 API（相容 + 新增）
- embed_clustering_leiden(...)
- knn_merge(...)
- strict_dedup(...)（相容 alias）
"""

import numpy as np
import time


# -----------------------------
# utils
# -----------------------------
def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _l2_normalize_rows(x, eps=1e-12):
    x = _as_numpy(x)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _try_import_igraph():
    try:
        import igraph as ig
        import leidenalg

        return ig, leidenalg
    except Exception:
        return None, None


# -----------------------------
# Step4: KNN similarity (cosine)
# -----------------------------
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

    torch = _try_import_torch() if use_torch else None
    if torch is not None:
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


def _drop_self_from_knn(knn_indices, knn_scores):
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


# -----------------------------
# Step5/6: Leiden（保留，不影響 merge）
# -----------------------------
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
        print("[edges] z_edge=%.3f kept %d / %d (%.2f%%)" % (z_edge, kept, total, 100.0 * kept / max(1, total)))

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
    ig, leidenalg = _try_import_igraph()
    if ig is None or leidenalg is None:
        raise ImportError("igraph/leidenalg not available. Please install python-igraph and leidenalg.")

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


def embed_clustering_leiden(
    embeddings,
    topk=150,
    chunk_size=20000,
    z_edge=0.5,
    mutual_boost=1.5,
    min_deg=2,
    min_deg_weight=0.8,
    resolutions=1.0,
    seed=0,
    return_intermediates=False,
    verbose=False,
):
    """
    embeddings -> KNN -> edge filter/weight -> Leiden -> cluster_id
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))
    N = X.shape[0]

    knn = compute_knn_similarity(
        X,
        topk=int(topk),
        chunk_size=int(chunk_size),
        ensure_normalized=True,
        use_torch=True,
        device="cuda",
        verbose=verbose,
    )

    edges0 = build_edges_from_knn(knn["indices"], knn["scores"], drop_self=True)
    nbr_idx = edges0["nbr_idx"]
    nbr_sim = edges0["nbr_sim"]

    nbr_idx2, nbr_sim2 = _drop_self_from_knn(nbr_idx, nbr_sim)

    edges = apply_edge_filter_and_weights(
        nbr_idx2,
        nbr_sim2,
        z_edge=float(z_edge),
        mutual_boost=float(mutual_boost),
        min_deg=int(min_deg) if min_deg is not None else None,
        min_deg_weight=float(min_deg_weight),
        verbose=verbose,
    )

    if isinstance(resolutions, (list, tuple, np.ndarray)):
        out = {}
        graphs = {}
        for r in resolutions:
            cid, g = leiden_cluster(
                edges["src"],
                edges["dst"],
                edges["sim"],
                n_nodes=N,
                resolution=float(r),
                seed=seed,
            )
            out[float(r)] = cid
            graphs[float(r)] = g
        if return_intermediates:
            inter = {"knn": knn, "edges": edges, "igraph": graphs}
            return out, inter
        return out

    cid, g = leiden_cluster(
        edges["src"],
        edges["dst"],
        edges["sim"],
        n_nodes=N,
        resolution=float(resolutions),
        seed=seed,
    )
    if return_intermediates:
        inter = {"knn": knn, "edges": edges, "igraph": g}
        return cid, inter
    return cid


# -----------------------------
# Union-Find (with optional size cap)
# -----------------------------
def _uf_init(n):
    parent = np.arange(int(n), dtype=np.int32)
    rank = np.zeros(int(n), dtype=np.int8)
    return parent, rank


def _uf_init_size(n, init_size=None):
    parent = np.arange(int(n), dtype=np.int32)
    rank = np.zeros(int(n), dtype=np.int8)
    if init_size is None:
        size = np.ones(int(n), dtype=np.int64)
    else:
        size = _as_numpy(init_size).astype(np.int64, copy=False)
        if int(size.shape[0]) != int(n):
            raise ValueError("init_size must be (N,), got %r" % (size.shape,))
    return parent, rank, size


def _uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent, rank, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return False
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1
    return True


def _uf_union_cap(parent, rank, size, a, b, cap):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return False

    if cap is not None:
        cap = int(cap)
        if size[ra] + size[rb] > cap:
            return False

    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    if rank[ra] == rank[rb]:
        rank[ra] += 1
    return True


def _compact_components(parent):
    n = int(parent.shape[0])
    roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = _uf_find(parent, i)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_id = inv.astype(np.int32)
    return comp_id, uniq.astype(np.int32)


# -----------------------------
# merge 核心：mutual top-k + sim（不使用 edge_gap）
# -----------------------------
def mutualk_merge_from_knn_sim(
    nbr_idx,
    nbr_sim,
    mutual_k=3,
    merge_top_p=1.0,
    merge_sim_min=None,
    merge_sim_quantile=None,
    sim_floor=None,
    shared_neighbor_min=0,
    shared_neighbor_topk=None,
    node_weight=None,
    max_cluster_weight=None,
    verbose=False,
):
    """
    mutual top-k + sim merge（不使用 edge_gap）

    規則：
      - i 的前 mutual_k 近鄰中包含 j
      - j 的前 mutual_k 近鄰中包含 i
      - similarity 門檻：
          * 若 merge_sim_min != None：使用固定門檻（相容舊行為）
          * 否則若 merge_sim_quantile != None：使用「mutual candidates 的 e_s」分位數當作門檻
      - 若 merge_top_p < 1：對通過門檻之候選邊依 sim 由大到小取前 p%

    max_cluster_weight:
      - 若提供，合併當下會限制 component weight 不得超過上限。
      - component weight 由 node_weight 決定；若 node_weight=None，則每個點 weight=1。
    """
    nbr_idx = _as_numpy(nbr_idx).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(nbr_sim).astype(np.float32, copy=False)
    if nbr_idx.ndim != 2 or nbr_sim.ndim != 2 or nbr_idx.shape != nbr_sim.shape:
        raise ValueError("nbr_idx/nbr_sim must be (N,K) with same shape")

    N, K = nbr_idx.shape

    mutual_k = int(mutual_k)
    if mutual_k <= 0:
        raise ValueError("mutual_k must be >= 1")
    kk = min(mutual_k, K)

    if kk <= 0:
        parent, rank = _uf_init(N)
        comp_id, roots = _compact_components(parent)
        return comp_id, {"n_candidates": 0, "n_edges_selected": 0, "n_unions": 0, "n_components": int(len(roots))}

    # 1) 收集 mutual candidates（不先做 sim 過濾）
    e_i = []
    e_j = []
    e_s = []

    for i in range(N):
        row_idx = nbr_idx[i, :kk]
        row_sim = nbr_sim[i, :kk]

        for pos in range(row_idx.shape[0]):
            j = int(row_idx[pos])
            if j < 0 or j >= N or j == i:
                continue

            rowj_idx = nbr_idx[j, :kk]
            is_mutual = False
            for t in range(rowj_idx.shape[0]):
                if int(rowj_idx[t]) == i:
                    is_mutual = True
                    break
            if not is_mutual:
                continue

            if i > j:
                continue

            e_i.append(int(i))
            e_j.append(int(j))
            e_s.append(float(row_sim[pos]))

    n_candidates = int(len(e_i))
    if verbose:
        print("[mutualk/sim] mutual_k=%d candidates=%d" % (kk, n_candidates))

    if n_candidates == 0:
        parent, rank = _uf_init(N)
        comp_id, roots = _compact_components(parent)
        return comp_id, {"n_candidates": 0, "n_edges_selected": 0, "n_unions": 0, "n_components": int(len(roots))}

    e_i = np.asarray(e_i, dtype=np.int32)
    e_j = np.asarray(e_j, dtype=np.int32)
    e_s = np.asarray(e_s, dtype=np.float32)

    # 2) 用 mutual candidates 的 e_s 分位數決定門檻（若未給 merge_sim_min）
    thr_base = None
    if merge_sim_min is None and merge_sim_quantile is not None:
        q = float(merge_sim_quantile)
        if not (0.0 < q < 1.0):
            raise ValueError("merge_sim_quantile must be in (0,1), got %r" % q)
        thr_base = float(np.quantile(e_s, q))
        merge_sim_min = thr_base
        if verbose:
            print(
                "[mutualk/sim] quantile=%.4f => merge_sim_min=%.6f (from %d mutual edges)"
                % (q, float(merge_sim_min), int(e_s.size))
            )
    elif merge_sim_min is not None:
        thr_base = float(merge_sim_min)

    thr_eff = None
    if sim_floor is None:
        thr_eff = thr_base
    elif thr_base is None:
        thr_eff = float(sim_floor)
    else:
        thr_eff = max(float(sim_floor), thr_base)

    # 3) 套用 sim 門檻（若有）
    if thr_eff is not None:
        keep = e_s >= float(thr_eff)
        e_i = e_i[keep]
        e_j = e_j[keep]
        e_s = e_s[keep]

    n_after_sim = int(e_s.size)
    k_sn = K if shared_neighbor_topk is None else int(shared_neighbor_topk)
    k_sn = max(1, min(k_sn, K))

    if e_s.size == 0:
        parent, rank = _uf_init(N)
        comp_id, roots = _compact_components(parent)
        return comp_id, {
            "n_candidates": int(n_candidates),
            "n_edges_selected": 0,
            "n_unions": 0,
            "n_components": int(len(roots)),
            "merge_sim_min": float(merge_sim_min) if merge_sim_min is not None else None,
            "merge_sim_quantile": float(merge_sim_quantile) if merge_sim_quantile is not None else None,
            "thr_base": float(thr_base) if thr_base is not None else None,
            "thr_eff": float(thr_eff) if thr_eff is not None else None,
            "sim_floor": float(sim_floor) if sim_floor is not None else None,
            "shared_neighbor_min": int(shared_neighbor_min),
            "shared_neighbor_topk": int(k_sn) if k_sn is not None else None,
            "n_after_sim": int(n_after_sim),
            "n_after_shared": 0,
        }

    # 3.5) 共同鄰居過濾

    if shared_neighbor_min > 0:
        cache = {}

        def _get_set(node):
            v = cache.get(node)
            if v is None:
                v = set(nbr_idx[node, :k_sn].tolist())
                cache[node] = v
            return v

        keep_shared = np.zeros_like(e_i, dtype=bool)
        for t in range(int(e_i.size)):
            ii = int(e_i[t])
            jj = int(e_j[t])
            set_i = _get_set(ii)
            set_j = _get_set(jj)
            if len(set_i) < len(set_j):
                cnt = sum(1 for x in set_i if x in set_j)
            else:
                cnt = sum(1 for x in set_j if x in set_i)
            if cnt >= int(shared_neighbor_min):
                keep_shared[t] = True
        e_i = e_i[keep_shared]
        e_j = e_j[keep_shared]
        e_s = e_s[keep_shared]
        n_after_shared = int(e_s.size)
    else:
        n_after_shared = int(e_s.size)

    p = float(merge_top_p)
    if not (0.0 < p <= 1.0):
        raise ValueError("merge_top_p must be in (0, 1], got %r" % p)

    if p < 1.0:
        m = int(max(1, round(e_s.size * p)))
        order = np.argsort(-e_s)[:m]
        sel_i = e_i[order]
        sel_j = e_j[order]
        sel_s = e_s[order]
        if verbose:
            print("[mutualk/sim] select top_p=%.4f => selected_edges=%d" % (p, int(sel_i.size)))
    else:
        sel_i, sel_j, sel_s = e_i, e_j, e_s
        if verbose:
            print("[mutualk/sim] select all => selected_edges=%d" % int(sel_i.size))

    if max_cluster_weight is None:
        parent, rank = _uf_init(N)
        n_unions = 0
        for t in range(int(sel_i.size)):
            if _uf_union(parent, rank, int(sel_i[t]), int(sel_j[t])):
                n_unions += 1
    else:
        parent, rank, size = _uf_init_size(N, init_size=node_weight)
        n_unions = 0
        for t in range(int(sel_i.size)):
            if _uf_union_cap(parent, rank, size, int(sel_i[t]), int(sel_j[t]), cap=max_cluster_weight):
                n_unions += 1

    comp_id, roots = _compact_components(parent)
    info = {
        "n_candidates": int(n_candidates),
        "n_edges_selected": int(sel_i.size),
        "n_unions": int(n_unions),
        "n_components": int(len(roots)),
        "selected_sim_mean": float(sel_s.mean()) if sel_s.size else 0.0,
        "selected_sim_min": float(sel_s.min()) if sel_s.size else 0.0,
        "selected_sim_max": float(sel_s.max()) if sel_s.size else 0.0,
        "merge_sim_min": float(merge_sim_min) if merge_sim_min is not None else None,
        "merge_sim_quantile": float(merge_sim_quantile) if merge_sim_quantile is not None else None,
        "thr_base": float(thr_base) if thr_base is not None else None,
        "thr_eff": float(thr_eff) if thr_eff is not None else None,
        "sim_floor": float(sim_floor) if sim_floor is not None else None,
        "shared_neighbor_min": int(shared_neighbor_min),
        "shared_neighbor_topk": int(k_sn) if k_sn is not None else None,
        "n_after_sim": int(n_after_sim),
        "n_after_shared": int(n_after_shared),
    }
    return comp_id, info


def aggregate_embeddings_by_component(embeddings, comp_id, ensure_normalized=True):
    """
    把同一個 component 內的 embedding 做平均，最後再 L2 normalize。
    回傳：
      X_merged: (n_comp, D)
      comp_size: (n_comp,)   # 這是「當前層節點數」，不是原始 spans 數
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if ensure_normalized:
        X = _l2_normalize_rows(X).astype(np.float32, copy=False)

    comp_id = _as_numpy(comp_id).astype(np.int32, copy=False)
    if comp_id.ndim != 1 or comp_id.shape[0] != X.shape[0]:
        raise ValueError("comp_id must be (N,), got %r" % (comp_id.shape,))

    n_comp = int(comp_id.max(initial=-1) + 1)
    if n_comp <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    D = int(X.shape[1])
    sums = np.zeros((n_comp, D), dtype=np.float32)
    np.add.at(sums, comp_id, X)
    cnt = np.bincount(comp_id, minlength=n_comp).astype(np.int32)
    cnt_safe = np.maximum(cnt, 1).astype(np.float32)
    X_merged = sums / cnt_safe[:, None]
    X_merged = _l2_normalize_rows(X_merged).astype(np.float32, copy=False)
    return X_merged, cnt


# -----------------------------
# 單輪 merge step（共用）
# -----------------------------
def merge_step_from_knn(
    embeddings,
    nbr_idx,
    nbr_sim,
    mutual_k=3,
    merge_top_p=0.01,
    merge_sim_min=None,
    merge_sim_quantile=None,
    sim_floor=None,
    shared_neighbor_min=0,
    shared_neighbor_topk=None,
    node_weight=None,
    max_cluster_weight=None,
    ensure_normalized=True,
    return_intermediates=False,
    verbose=False,
):
    """
    已經有 KNN（nbr_idx/nbr_sim）時，用同一個核心做 merge + 聚合。

    回傳：
      comp_id, X_merged, comp_size, info
      (option) inter: {"nbr_idx":..., "nbr_sim":...}
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))

    nbr_idx = _as_numpy(nbr_idx).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(nbr_sim).astype(np.float32, copy=False)
    if nbr_idx.ndim != 2 or nbr_sim.ndim != 2 or nbr_idx.shape != nbr_sim.shape:
        raise ValueError("nbr_idx/nbr_sim must be (N,K) with same shape")
    if nbr_idx.shape[0] != X.shape[0]:
        raise ValueError("nbr_idx first dim must match embeddings N")

    comp_id, merge_info = mutualk_merge_from_knn_sim(
        nbr_idx,
        nbr_sim,
        mutual_k=int(mutual_k),
        merge_top_p=float(merge_top_p),
        merge_sim_min=float(merge_sim_min) if merge_sim_min is not None else None,
        merge_sim_quantile=float(merge_sim_quantile) if merge_sim_quantile is not None else None,
        sim_floor=float(sim_floor) if sim_floor is not None else None,
        shared_neighbor_min=int(shared_neighbor_min),
        shared_neighbor_topk=shared_neighbor_topk if shared_neighbor_topk is None else int(shared_neighbor_topk),
        node_weight=node_weight,
        max_cluster_weight=max_cluster_weight,
        verbose=verbose,
    )

    X_merged, comp_size = aggregate_embeddings_by_component(X, comp_id, ensure_normalized=ensure_normalized)

    if return_intermediates:
        inter = {"nbr_idx": nbr_idx, "nbr_sim": nbr_sim}
        return comp_id, X_merged, comp_size, merge_info, inter
    return comp_id, X_merged, comp_size, merge_info


def merge_step_from_embeddings(
    embeddings,
    mutual_k=3,
    merge_knn_topk=50,
    merge_top_p=0.01,
    merge_sim_min=None,
    merge_sim_quantile=None,
    sim_floor=None,
    shared_neighbor_min=0,
    shared_neighbor_topk=None,
    chunk_size=20000,
    node_weight=None,
    max_cluster_weight=None,
    ensure_normalized=True,
    return_intermediates=False,
    verbose=False,
):
    """
    單輪 merge step（最常用）：
      embeddings -> KNN -> drop_self -> mutual-k + sim merge -> 聚合
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))
    N = int(X.shape[0])
    if N == 0:
        comp_id = np.zeros((0,), dtype=np.int32)
        X_merged = np.zeros((0, X.shape[1]), dtype=np.float32)
        comp_size = np.zeros((0,), dtype=np.int32)
        info = {"n_candidates": 0, "n_edges_selected": 0, "n_unions": 0, "n_components": 0}
        if return_intermediates:
            return comp_id, X_merged, comp_size, info, {"knn": None}
        return comp_id, X_merged, comp_size, info

    mutual_k = int(mutual_k)
    if mutual_k < 1:
        raise ValueError("mutual_k must be >= 1")

    mk = int(merge_knn_topk)
    mk = max(mk, mutual_k + 2)
    mk = min(mk, max(2, N - 1))

    knn = compute_knn_similarity(
        X,
        topk=mk,
        chunk_size=int(chunk_size),
        ensure_normalized=True,
        use_torch=True,
        device="cuda",
        verbose=verbose,
    )

    # 直接用 (N,K) 的 knn，去掉 self（通常是第 0 欄）
    nbr_idx, nbr_sim = _drop_self_from_knn(knn["indices"], knn["scores"])

    out = merge_step_from_knn(
        X,
        nbr_idx,
        nbr_sim,
        mutual_k=mutual_k,
        merge_top_p=merge_top_p,
        merge_sim_min=merge_sim_min,
        merge_sim_quantile=merge_sim_quantile,
        sim_floor=sim_floor,
        shared_neighbor_min=shared_neighbor_min,
        shared_neighbor_topk=shared_neighbor_topk,
        node_weight=node_weight,
        max_cluster_weight=max_cluster_weight,
        ensure_normalized=ensure_normalized,
        return_intermediates=return_intermediates,
        verbose=verbose,
    )

    if return_intermediates:
        comp_id, X_merged, comp_size, info, inter = out
        inter["knn"] = knn
        return comp_id, X_merged, comp_size, info, inter

    return out


# -----------------------------
# knn_merge（多輪 iterative merge）
# -----------------------------
def knn_merge(
    embeddings,
    mutual_k=3,
    merge_knn_topk=3,
    merge_top_p=1.0,
    merge_sim_min=0.92,
    merge_sim_quantile=None,
    sim_floor=None,
    shared_neighbor_min=0,
    shared_neighbor_topk=None,
    chunk_size=20000,
    max_iters=10,
    min_improve=0.001,
    max_cluster_size=100,
    node_weight=None,
    ensure_normalized=True,
    rep_chunk_size=5000,
    return_history=True,
    verbose=True,
):
    """
    iterative mutual-k + sim merge 做嚴格去重，並回傳每群代表 rep_index。

    Input:
      embeddings: (N, D)
      node_weight: (N,) 可選（原始權重），用於 cap（max_cluster_size）。
                   若直接對原始 spans 做 dedup，可不給（預設全 1）。

      merge_sim_min:
        - 固定門檻（相容舊行為）
      merge_sim_quantile:
        - 若 merge_sim_min=None 且 merge_sim_quantile != None，則以「mutual candidates 的 e_s」分位數決定門檻
        - 例如 0.90 代表使用 90th percentile（只保留最上面 10% 的 mutual 邊）

    Output:
      cluster_id:   (N0,) int32，原始點 -> 最終 cluster id (0..n_clusters-1)
      X_final:      (n_clusters, D) float32，最終 cluster embedding（mean + L2）
      cluster_size: (n_clusters,) int64，每 cluster 的原始點數（應 <= max_cluster_size）
      rep_index:    (n_clusters,) int64，每 cluster 代表的「原始 row index」
      info: dict（final summary + 每輪 history）
    """
    X0 = np.asarray(embeddings)
    if X0.ndim != 2:
        raise ValueError("embeddings must be (N,D), got %r" % (X0.shape,))
    X0 = X0.astype(np.float32, copy=False)

    N0, D = X0.shape
    if N0 == 0:
        cluster_id = np.zeros((0,), dtype=np.int32)
        X_final = np.zeros((0, D), dtype=np.float32)
        cluster_size = np.zeros((0,), dtype=np.int64)
        rep_index = np.zeros((0,), dtype=np.int64)
        info = {"history": [], "final": {"N0": 0, "n_clusters": 0}}
        return cluster_id, X_final, cluster_size, rep_index, info

    X_cur = _l2_normalize_rows(X0).astype(np.float32) if ensure_normalized else X0

    # original -> current mapping
    orig_to_cur = np.arange(N0, dtype=np.int32)

    # current node weight（用於 cap）
    if node_weight is None:
        w_cur = np.ones((X_cur.shape[0],), dtype=np.int64)
    else:
        w = np.asarray(node_weight)
        if w.shape[0] != N0:
            raise ValueError("node_weight must have length %d, got %d" % (N0, w.shape[0]))
        w_cur = w.astype(np.int64, copy=False)

    prev_n = int(X_cur.shape[0])
    history = []

    for it in range(int(max_iters)):
        t0 = time.time()
        if ensure_normalized:
            X_cur = _l2_normalize_rows(X_cur).astype(np.float32)

        if verbose:
            print("\n== iter", it, "==")

        comp_id, X_next, comp_size_cur, merge_info = merge_step_from_embeddings(
            X_cur,
            mutual_k=int(mutual_k),
            merge_knn_topk=int(merge_knn_topk),
            merge_top_p=float(merge_top_p),
            merge_sim_min=float(merge_sim_min) if merge_sim_min is not None else None,
            merge_sim_quantile=float(merge_sim_quantile) if merge_sim_quantile is not None else None,
            sim_floor=float(sim_floor) if sim_floor is not None else None,
            shared_neighbor_min=int(shared_neighbor_min),
            shared_neighbor_topk=shared_neighbor_topk if shared_neighbor_topk is None else int(shared_neighbor_topk),
            chunk_size=int(chunk_size),
            node_weight=w_cur,
            max_cluster_weight=int(max_cluster_size),
            ensure_normalized=True,
            return_intermediates=False,
            verbose=bool(verbose),
        )

        # 更新 original -> current
        orig_to_cur = comp_id[orig_to_cur]

        new_n = int(X_next.shape[0])
        step_improve = (prev_n - new_n) / max(1, prev_n)
        overall = new_n / max(1, N0)

        # 更新下一層 cap weight
        w_next = np.bincount(comp_id, weights=w_cur, minlength=new_n).astype(np.int64)
        w_cur = w_next

        # 原始點角度的 cluster size（應 <= cap）
        sizes_orig = np.bincount(orig_to_cur, minlength=new_n).astype(np.int64)

        rec = {
            "iter": int(it),
            "prev_n": int(prev_n),
            "new_n": int(new_n),
            "step_improve": float(step_improve),
            "overall": float(overall),
            "max_size_current_layer": int(comp_size_cur.max()) if comp_size_cur.size else 0,
            "max_size_original": int(sizes_orig.max()) if sizes_orig.size else 0,
            "weight_max_next": int(w_cur.max()) if w_cur.size else 0,
            "time_sec": float(time.time() - t0),
        }
        for k, v in (merge_info or {}).items():
            if k not in rec:
                rec[k] = v

        if verbose:
            print(
                "n_comp:",
                new_n,
                "max_size_current_layer:",
                rec["max_size_current_layer"],
                "n_candidates:",
                int(rec.get("n_candidates", 0)),
                "n_edges_selected:",
                int(rec.get("n_edges_selected", 0)),
                "n_unions:",
                int(rec.get("n_unions", 0)),
            )
            print(
                "[iter %d] prev_n=%d -> new_n=%d  step_improve=%.4f  overall=%.4f"
                % (it, prev_n, new_n, step_improve, overall)
            )
            print(
                "  ORIGINAL max_cluster_size:",
                int(sizes_orig.max()),
                "(cap=%d)" % int(max_cluster_size),
            )
            print("  time:", rec["time_sec"])

        if return_history:
            history.append(rec)

        # stop 條件
        if new_n == prev_n:
            if verbose:
                print("Stop: no change in n_comp")
            X_cur = X_next
            break
        if int(rec.get("n_unions", 0)) == 0:
            if verbose:
                print("Stop: no successful unions (blocked by cap or thresholds)")
            X_cur = X_next
            break
        if step_improve < float(min_improve):
            if verbose:
                print("Stop: improvement too small")
            X_cur = X_next
            break

        X_cur = X_next
        prev_n = new_n

    # final outputs
    cluster_id = orig_to_cur.astype(np.int32, copy=False)
    X_final = X_cur.astype(np.float32, copy=False)

    n_clusters = int(X_final.shape[0])
    cluster_size = np.bincount(cluster_id, minlength=n_clusters).astype(np.int64)

    # ---- rep_index：每群選一個「最接近群 embedding」的原始 index（cosine）----
    X0n = _l2_normalize_rows(X0).astype(np.float32)
    Xcn = _l2_normalize_rows(X_final).astype(np.float32)

    scores = np.empty((N0,), dtype=np.float32)
    bsz = int(rep_chunk_size)
    for st in range(0, N0, bsz):
        ed = min(N0, st + bsz)
        c = cluster_id[st:ed]
        scores[st:ed] = np.sum(X0n[st:ed] * Xcn[c], axis=1)

    # cluster_id 為主鍵（分群），-scores 為次鍵（分群內高分在前）
    order = np.lexsort((-scores, cluster_id))
    cid_sorted = cluster_id[order]
    idx_sorted = np.arange(N0, dtype=np.int64)[order]

    first = np.ones_like(cid_sorted, dtype=bool)
    first[1:] = cid_sorted[1:] != cid_sorted[:-1]

    rep_index = np.full((n_clusters,), -1, dtype=np.int64)
    rep_index[cid_sorted[first]] = idx_sorted[first]

    if np.any(rep_index < 0):
        miss = np.where(rep_index < 0)[0]
        for cid in miss.tolist():
            rep_index[cid] = int(np.where(cluster_id == cid)[0][0])

    info = {
        "history": history,
        "final": {
            "N0": int(N0),
            "D": int(D),
            "n_clusters": int(n_clusters),
            "cluster_size_min": int(cluster_size.min()) if cluster_size.size else 0,
            "cluster_size_median": int(np.median(cluster_size)) if cluster_size.size else 0,
            "cluster_size_mean": float(cluster_size.mean()) if cluster_size.size else 0.0,
            "cluster_size_max": int(cluster_size.max()) if cluster_size.size else 0,
            "n_violations_over_cap": int((cluster_size > int(max_cluster_size)).sum()),
            "cap": int(max_cluster_size),
            "merge_sim_min": float(merge_sim_min) if merge_sim_min is not None else None,
            "merge_sim_quantile": float(merge_sim_quantile) if merge_sim_quantile is not None else None,
            "sim_floor": float(sim_floor) if sim_floor is not None else None,
            "shared_neighbor_min": int(shared_neighbor_min),
            "shared_neighbor_topk": int(shared_neighbor_topk) if shared_neighbor_topk is not None else None,
            "mutual_k": int(mutual_k),
            "merge_knn_topk": int(merge_knn_topk),
            "merge_top_p": float(merge_top_p),
        },
    }

    return cluster_id, X_final, cluster_size, rep_index, info


def strict_dedup(*args, **kwargs):
    import warnings

    warnings.warn("strict_dedup is deprecated; use knn_merge instead", DeprecationWarning, stacklevel=2)
    return knn_merge(*args, **kwargs)

