#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_clustering.py（legacy 風格，無 typing / 無 dataclass）

把 Step4 / Step5 / Step6 包成「乾淨、可重用」的流程：

Input  : embeddings (N, D)  —— span embedding（建議已做 L2 normalize；未做也可，內部可補）
Output : cluster_id (N,)    —— 每個 span 的群集編號（Leiden）

設計原則
- 不做任何資料庫 / parquet 寫入（I/O 完全拆開）
- 流程標準化：embedding -> KNN -> edge 篩選/加權 -> Leiden -> cluster index
- GPU 可用時用 torch 加速；否則自動退回 CPU numpy
- 依賴：numpy；（選用 torch）；igraph + leidenalg

公開 API
- embed_clustering_leiden(...)          : 原本 embed_clustering 改名
- embed_clustering_mutualk_merge(...)   : 只做 mutual top-k + edge_gap 合併（不跑 Leiden）
"""

import numpy as np


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


def _l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


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
def compute_knn_similarity(embeddings, topk=150, chunk_size=20000, ensure_normalized=True,
                           use_torch=True, device="cuda", verbose=False):
    """
    回傳：
      indices: (N, topk) int32
      scores : (N, topk) float32
    備註：topk 通常會包含自己（第一名），後續建邊時會丟掉 self-loop。
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))

    if ensure_normalized:
        X = _l2_normalize(X)

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
                chunk = X_t[start:end]          # (B, D)
                sims = chunk @ X_t.T            # (B, N)
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
        chunk = X[start:end]                  # (B, D)
        sims = np.matmul(chunk, Xt)           # (B, N)

        part = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]  # (B, topk)
        part_sims = np.take_along_axis(sims, part, axis=1)             # (B, topk)
        order = np.argsort(-part_sims, axis=1)
        idxs = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(part_sims, order, axis=1)

        all_idxs[start:end] = idxs.astype(np.int32)
        all_vals[start:end] = vals.astype(np.float32)

    return {"indices": all_idxs, "scores": all_vals}


# -----------------------------
# Step5: build edges + filter/weight
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


def apply_edge_filter_and_weights(nbr_idx, nbr_sim, z_edge=0.5, mutual_boost=1.5,
                                 min_deg=2, min_deg_weight=0.8, verbose=False):
    """
    對齊 step5 的概念：
    - global z-score（用所有 sim 的 mean/std）→ mask = (z >= z_edge)
    - mutual kNN 的邊：只做加權（sim * mutual_boost），不作硬過濾
    - 最低出度修補：若某點保留邊數 < min_deg，從原始 KNN 的前幾名補回（並把權重打折）
    """
    nbr_idx = _as_numpy(nbr_idx).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(nbr_sim).astype(np.float32, copy=False)
    N, K1 = nbr_idx.shape

    flat = nbr_sim.reshape(-1).astype(np.float32)
    mu = float(flat.mean())
    sigma = float(flat.std()) + 1e-8
    z = (nbr_sim - mu) / sigma
    mask = (z >= z_edge)

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


# -----------------------------
# Step6: Leiden clustering
# -----------------------------
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
            seed=int(seed)
        )
    except TypeError:
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=float(resolution)
        )

    cluster_id = np.zeros(int(n_nodes), dtype=np.int32)
    for cid, nodes in enumerate(partition):
        for v in nodes:
            cluster_id[int(v)] = int(cid)

    return cluster_id, g


# -----------------------------
# Union-Find (for mutual-k merge)
# -----------------------------
def _uf_init(n):
    parent = np.arange(int(n), dtype=np.int32)
    rank = np.zeros(int(n), dtype=np.int8)
    return parent, rank


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


def _compact_components(parent):
    n = int(parent.shape[0])
    roots = np.empty(n, dtype=np.int32)
    for i in range(n):
        roots[i] = _uf_find(parent, i)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_id = inv.astype(np.int32)
    return comp_id, uniq.astype(np.int32)


# -----------------------------
# mutual top-k + edge_gap merge (from KNN)
# -----------------------------
def mutualk_merge_from_knn(nbr_idx, nbr_sim, mutual_k=3,
                          merge_top_p=0.01,
                          merge_gap_min=None,
                          merge_sim_min=None,
                          verbose=False):
    """
    用 mutual top-k + edge_gap 做合併。

    輸入
      nbr_idx: (N, K)  已去除 self 的 knn indices（由近到遠）
      nbr_sim: (N, K)  對應相似度

    edge_gap 定義（針對這條邊）：
      edge_gap(i,j) = sim(i,j) - max( alt_i(excluding j), alt_j(excluding i) )
      其中 alt 是在 top-k 內（排除對方）最大的替代相似度。
      kk==1 時沒有替代者，這裡用 edge_gap=sim(i,j) 避免不合理的巨大 gap。

    回傳
      comp_id: (N,) 每個點的 component id（compact）
      info: dict（候選邊數、採用邊數、component 數等）
    """
    nbr_idx = _as_numpy(nbr_idx).astype(np.int32, copy=False)
    nbr_sim = _as_numpy(nbr_sim).astype(np.float32, copy=False)
    N, K = nbr_idx.shape

    mutual_k = int(mutual_k)
    if mutual_k <= 0:
        raise ValueError("mutual_k must be >= 1")
    kk = min(mutual_k, K)

    if kk <= 0:
        parent, rank = _uf_init(N)
        comp_id, roots = _compact_components(parent)
        return comp_id, {"n_candidates": 0, "n_selected": 0, "n_components": int(len(roots))}

    e_i = []
    e_j = []
    e_sim = []
    e_gap = []

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

            sim_ij = float(row_sim[pos])

            if kk == 1:
                edge_gap = sim_ij
            else:
                alt_i = -1e9
                for t in range(kk):
                    if t == pos:
                        continue
                    v = float(row_sim[t])
                    if v > alt_i:
                        alt_i = v

                alt_j = -1e9
                rowj_sim = nbr_sim[j, :kk]
                for t in range(kk):
                    if int(rowj_idx[t]) == i:
                        continue
                    v = float(rowj_sim[t])
                    if v > alt_j:
                        alt_j = v

                edge_gap = sim_ij - max(alt_i, alt_j)

            if merge_sim_min is not None and sim_ij < float(merge_sim_min):
                continue
            if merge_gap_min is not None and edge_gap < float(merge_gap_min):
                continue

            e_i.append(int(i))
            e_j.append(int(j))
            e_sim.append(float(sim_ij))
            e_gap.append(float(edge_gap))

    n_candidates = int(len(e_i))
    if verbose:
        print("[mutualk] mutual_k=%d candidates=%d" % (kk, n_candidates))

    if n_candidates == 0:
        parent, rank = _uf_init(N)
        comp_id, roots = _compact_components(parent)
        return comp_id, {"n_candidates": 0, "n_selected": 0, "n_components": int(len(roots))}

    e_i = np.asarray(e_i, dtype=np.int32)
    e_j = np.asarray(e_j, dtype=np.int32)
    e_gap = np.asarray(e_gap, dtype=np.float32)

    if merge_gap_min is None:
        p = float(merge_top_p)
        if not (0.0 < p <= 1.0):
            raise ValueError("merge_top_p must be in (0, 1], got %r" % p)

        good = e_gap > 0.0
        e_i2 = e_i[good]
        e_j2 = e_j[good]
        e_gap2 = e_gap[good]

        if e_gap2.size == 0:
            parent, rank = _uf_init(N)
            comp_id, roots = _compact_components(parent)
            return comp_id, {"n_candidates": n_candidates, "n_selected": 0, "n_components": int(len(roots))}

        m = int(max(1, round(e_gap2.size * p)))
        order = np.argsort(-e_gap2)[:m]
        sel_i = e_i2[order]
        sel_j = e_j2[order]
        sel_gap = e_gap2[order]

        if verbose:
            print("[mutualk] select top_p=%.4f => selected=%d (from gap>0 edges=%d)" %
                  (p, int(sel_i.size), int(e_gap2.size)))
    else:
        sel_i = e_i
        sel_j = e_j
        sel_gap = e_gap
        if verbose:
            print("[mutualk] using hard gap_min => selected=%d" % int(sel_i.size))

    parent, rank = _uf_init(N)
    n_selected = 0
    for t in range(int(sel_i.size)):
        if _uf_union(parent, rank, int(sel_i[t]), int(sel_j[t])):
            n_selected += 1

    comp_id, roots = _compact_components(parent)
    info = {
        "n_candidates": int(n_candidates),
        "n_selected": int(n_selected),
        "n_components": int(len(roots)),
        "selected_edge_gap_mean": float(sel_gap.mean()) if sel_gap.size > 0 else 0.0,
        "selected_edge_gap_min": float(sel_gap.min()) if sel_gap.size > 0 else 0.0,
        "selected_edge_gap_max": float(sel_gap.max()) if sel_gap.size > 0 else 0.0,
    }
    return comp_id, info


def aggregate_embeddings_by_component(embeddings, comp_id, ensure_normalized=True):
    """
    把同一個 component 內的 embedding 做平均，最後再 L2 normalize。
    回傳：
      X_merged: (n_comp, D)
      comp_size: (n_comp,)
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if ensure_normalized:
        X = _l2_normalize(X)

    comp_id = _as_numpy(comp_id).astype(np.int32, copy=False)
    n_comp = int(comp_id.max(initial=-1) + 1)
    if n_comp <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    D = int(X.shape[1])
    sums = np.zeros((n_comp, D), dtype=np.float32)
    np.add.at(sums, comp_id, X)
    cnt = np.bincount(comp_id, minlength=n_comp).astype(np.int32)
    cnt_safe = np.maximum(cnt, 1).astype(np.float32)
    X_merged = sums / cnt_safe[:, None]
    X_merged = _l2_normalize(X_merged)
    return X_merged.astype(np.float32, copy=False), cnt


# -----------------------------
# Public API: embed_clustering_leiden (原 embed_clustering 改名)
# -----------------------------
def embed_clustering_leiden(embeddings,
                           topk=150,
                           chunk_size=20000,
                           z_edge=0.5,
                           mutual_boost=1.5,
                           min_deg=2,
                           min_deg_weight=0.8,
                           resolutions=1.0,
                           seed=0,
                           return_intermediates=False,
                           verbose=False):
    """
    完整流程：
      embeddings -> KNN -> edge filter/weight -> Leiden -> cluster_id

    resolutions:
      - float/int：單一解析度，回傳 (N,)
      - list/tuple：多解析度，回傳 dict {res: (N,)}
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
        verbose=verbose
    )

    edges0 = build_edges_from_knn(knn["indices"], knn["scores"], drop_self=True)
    nbr_idx = edges0["nbr_idx"]
    nbr_sim = edges0["nbr_sim"]

    # 常見情況 self 在第 0 欄，直接丟掉即可（與你原流程對齊）
    if nbr_idx.shape[1] >= 2:
        nbr_idx2 = nbr_idx[:, 1:]
        nbr_sim2 = nbr_sim[:, 1:]
    else:
        nbr_idx2 = nbr_idx
        nbr_sim2 = nbr_sim

    edges = apply_edge_filter_and_weights(
        nbr_idx2,
        nbr_sim2,
        z_edge=float(z_edge),
        mutual_boost=float(mutual_boost),
        min_deg=int(min_deg) if min_deg is not None else None,
        min_deg_weight=float(min_deg_weight),
        verbose=verbose
    )

    if isinstance(resolutions, (list, tuple, np.ndarray)):
        out = {}
        graphs = {}
        for r in resolutions:
            cid, g = leiden_cluster(edges["src"], edges["dst"], edges["sim"],
                                    n_nodes=N, resolution=float(r), seed=seed)
            out[float(r)] = cid
            graphs[float(r)] = g
        if return_intermediates:
            inter = {"knn": knn, "edges": edges, "igraph": graphs}
            return out, inter
        return out

    cid, g = leiden_cluster(edges["src"], edges["dst"], edges["sim"],
                            n_nodes=N, resolution=float(resolutions), seed=seed)
    if return_intermediates:
        inter = {"knn": knn, "edges": edges, "igraph": g}
        return cid, inter
    return cid


# -----------------------------
# Public API: embed_clustering_mutualk_merge (merge-only)
# -----------------------------
def embed_clustering_mutualk_merge(embeddings,
                                  mutual_k=3,
                                  merge_knn_topk=50,
                                  merge_top_p=0.01,
                                  merge_gap_min=None,
                                  merge_sim_min=None,
                                  chunk_size=20000,
                                  return_intermediates=False,
                                  verbose=False):
    """
    只做 mutual top-k + edge_gap 的合併（近重複壓縮），不跑 Leiden。

    Input:
      embeddings: (N, D)

    Output:
      comp_id:    (N,)    每個原始點的 component id（0..n_comp-1）
      X_merged:   (n_comp, D) 聚合後的 embedding（mean + L2 normalize）
      comp_size:  (n_comp,)   每個 component 的點數
      info: dict           合併統計（候選邊數、採用邊數、component 數等）

    參數:
      mutual_k:
        mutual top-k 的 k（建議 3）
      merge_knn_topk:
        合併階段用的 knn topk（要 >= mutual_k+2；越大候選越穩但較慢）
      merge_top_p:
        若 merge_gap_min=None，則用 edge_gap 排名前 p% 的 mutual 邊做合併（避免硬選門檻）
      merge_gap_min / merge_sim_min:
        可選硬門檻（若給 merge_gap_min，就不走 top_p）
    """
    X = _as_numpy(embeddings).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N, D), got %r" % (X.shape,))
    N = int(X.shape[0])
    if N == 0:
        comp_id = np.zeros((0,), dtype=np.int32)
        X_merged = np.zeros((0, X.shape[1]), dtype=np.float32)
        comp_size = np.zeros((0,), dtype=np.int32)
        info = {"n_candidates": 0, "n_selected": 0, "n_components": 0}
        if return_intermediates:
            return comp_id, X_merged, comp_size, info, {"knn_merge": None}
        return comp_id, X_merged, comp_size, info

    mutual_k = int(mutual_k)
    if mutual_k < 1:
        raise ValueError("mutual_k must be >= 1")

    mk = int(merge_knn_topk)
    mk = max(mk, mutual_k + 2)          # +2 是為了更保險（含 self 的情況下）
    mk = min(mk, max(2, N - 1))

    # (A) KNN for merge
    knn_m = compute_knn_similarity(
        X,
        topk=mk,
        chunk_size=int(chunk_size),
        ensure_normalized=True,
        use_torch=True,
        device="cuda",
        verbose=verbose
    )

    # (B) drop self（沿用你檔案中的寫法：假設 self 多半在第 0 欄，所以用 [:,1:]）
    edges0 = build_edges_from_knn(knn_m["indices"], knn_m["scores"], drop_self=True)
    nbr_idx_full = edges0["nbr_idx"]
    nbr_sim_full = edges0["nbr_sim"]

    if nbr_idx_full.shape[1] >= 2:
        nbr_idx2 = nbr_idx_full[:, 1:]
        nbr_sim2 = nbr_sim_full[:, 1:]
    else:
        nbr_idx2 = nbr_idx_full
        nbr_sim2 = nbr_sim_full

    # (C) mutual-k merge
    comp_id, merge_info = mutualk_merge_from_knn(
        nbr_idx2,
        nbr_sim2,
        mutual_k=mutual_k,
        merge_top_p=merge_top_p,
        merge_gap_min=merge_gap_min,
        merge_sim_min=merge_sim_min,
        verbose=verbose
    )

    # (D) aggregate embeddings
    X_merged, comp_size = aggregate_embeddings_by_component(X, comp_id, ensure_normalized=True)

    if verbose:
        n_comp = int(X_merged.shape[0])
        print("[mutualk] N=%d -> n_comp=%d (compression=%.3f)" %
              (N, n_comp, float(n_comp) / max(1, N)))

    if return_intermediates:
        inter = {"knn_merge": knn_m}
        return comp_id, X_merged, comp_size, merge_info, inter

    return comp_id, X_merged, comp_size, merge_info
