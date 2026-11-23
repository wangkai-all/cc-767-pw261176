"""连续空间块划分与块交叉验证。

要点：
- 验证按连续空间块（而非随机点）划分，避免相邻点泄漏造成精度高估。
- 同批重复采样保持在一起：划分单元是"批次"（有批次号时），
  同批测点必属同一块 / 同一折。
- 支持用户在网页上画矩形验证块：每个矩形即一折的测试集。
"""
from __future__ import annotations

import numpy as np


def _kmeans(xy, k, seed=0, iters=60):
    """轻量 Lloyd 聚类（避免依赖 sklearn）。"""
    rng = np.random.default_rng(seed)
    n = len(xy)
    k = max(1, min(k, n))
    # kmeans++ 初始化
    centers = [xy[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min(((xy[:, None, :] - np.array(centers)[None])**2)
                    .sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        centers.append(xy[rng.choice(n, p=probs)])
    C = np.array(centers)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((xy[:, None, :] - C[None])**2).sum(-1)
        new = d2.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = xy[lab == j].mean(0)
    return lab


def assign_blocks(coords, batches, n_blocks, seed=0):
    """把测点分到连续空间块；同批测点不拆分。

    返回每个测点的块号 (n,) int 数组。
    """
    coords = np.asarray(coords, dtype=float)
    batches = np.asarray(batches, dtype=object)
    uniq = list(dict.fromkeys(batches.tolist()))
    if len(uniq) == 1 and str(uniq[0]) in ("", "None"):
        return _kmeans(coords, n_blocks, seed)
    # 以批次为单位聚类批次质心
    cent = np.array([coords[batches == b].mean(0) for b in uniq])
    bloc = _kmeans(cent, min(n_blocks, len(uniq)), seed)
    b2block = {b: int(bloc[i]) for i, b in enumerate(uniq)}
    return np.array([b2block[b] for b in batches], dtype=int)


def blocks_from_rects(coords, rects):
    """用户画的矩形验证块：rects = [[x0,y0,x1,y1],...]。

    返回 (block_ids, covered)：落在任一矩形内的点归对应块（测试折），
    其余点块号为 -1（始终作为训练）。
    """
    coords = np.asarray(coords, dtype=float)
    ids = np.full(len(coords), -1, dtype=int)
    for k, (x0, y0, x1, y1) in enumerate(rects):
        lo = np.array([min(x0, x1), min(y0, y1)])
        hi = np.array([max(x0, x1), max(y0, y1)])
        inside = ((coords >= lo) & (coords <= hi)).all(1)
        ids[inside] = k
    return ids


def make_folds(block_ids, n_folds=None):
    """由块号构造折：每折 = 若干块的并集。块号 -1 表示永远训练。"""
    uniq = sorted(set(int(b) for b in block_ids if b >= 0))
    if not uniq:
        return []
    if n_folds is None or n_folds >= len(uniq):
        return [{ "test_blocks": [u]} for u in uniq]
    folds = []
    for f in range(n_folds):
        tb = [u for i, u in enumerate(uniq) if i % n_folds == f]
        folds.append({"test_blocks": tb})
    return folds


def cross_validate(coords, covs, values, batches, counts, fit_fn,
                   block_ids, target="measurement"):
    """执行块交叉验证。

    fit_fn(train_idx, test_idx) -> dict(mean, var)  对测试点的预测。
    参数估计发生在 fit_fn 内部（只用训练块）。
    返回每折与总体指标。
    """
    values = np.asarray(values, dtype=float)
    folds = make_folds(block_ids)
    per_fold, all_err, all_z, all_cov90, all_cov95, all_var = \
        [], [], [], [], [], []
    for fi, fold in enumerate(folds):
        test_mask = np.isin(block_ids, fold["test_blocks"])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) < 8:
            continue
        pred = fit_fn(train_idx, test_idx)
        mean, var = np.asarray(pred["mean"]), np.asarray(pred["var"])
        err = values[test_idx] - mean
        sd = np.sqrt(np.maximum(var, 1e-15))
        z = err / sd
        cov90 = (np.abs(z) <= 1.6448536269514722).mean()
        cov95 = (np.abs(z) <= 1.959963984540054).mean()
        per_fold.append({
            "fold": fi, "n_test": int(len(test_idx)),
            "n_train": int(len(train_idx)),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "coverage90": float(cov90), "coverage95": float(cov95),
            "mean_z": float(z.mean()), "rms_z": float(np.sqrt(np.mean(z**2))),
            "params": pred.get("params"),
        })
        all_err.append(err); all_z.append(z)
        all_cov90.append(np.abs(z) <= 1.6448536269514722)
        all_cov95.append(np.abs(z) <= 1.959963984540054)
        all_var.append(var)
    if not all_err:
        return {"folds": [], "overall": None}
    err = np.concatenate(all_err)
    z = np.concatenate(all_z)
    overall = {
        "n": int(len(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "coverage90": float(np.concatenate(all_cov90).mean()),
        "coverage95": float(np.concatenate(all_cov95).mean()),
        "rms_z": float(np.sqrt(np.mean(z**2))),
    }
    return {"folds": per_fold, "overall": overall}


def random_split_ids(n, n_folds, seed=0):
    """随机拆分（对照组，展示精度高估）。返回 block_ids 形状 (n,)。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n)
