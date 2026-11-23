"""数据预处理：重复坐标、近重合点、数值病态 —— 三类问题分别处理。

每次处理都记录"改变了什么"，返回报告对象供界面展示：

- 精确重复坐标：合并为单个测点（取均值，测量次数 m 使块金降为 tau^2/m）。
- 近重合点（距离 < eps）：按策略 merge（合并到质心）或 jitter（确定性微扰）。
- 数值病态：协方差矩阵 Cholesky 失败或条件数超阈时，逐级加对角 jitter，
  报告 jitter 大小与条件数变化。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PreprocessReport:
    merges: list = field(default_factory=list)      # 精确重复合并
    near: list = field(default_factory=list)        # 近重合处理
    dropped: list = field(default_factory=list)     # 丢弃（如落入屏障）
    conditioning: dict = field(default_factory=dict)  # 病态加噪
    trend: dict = field(default_factory=dict)       # 趋势列处理

    def to_dict(self):
        return {
            "merges": self.merges,
            "near_duplicates": self.near,
            "dropped": self.dropped,
            "conditioning": self.conditioning,
            "trend": self.trend,
        }

    @property
    def changed(self) -> bool:
        return bool(self.merges or self.near or self.dropped
                    or self.conditioning.get("jitter"))


def merge_exact_duplicates(coords, values, covs, batches, ids, report):
    """合并坐标完全相同的测点。返回新数组（含 counts）。"""
    n = len(values)
    key_to_group: dict[tuple, list[int]] = {}
    for i in range(n):
        key_to_group.setdefault((float(coords[i, 0]), float(coords[i, 1])),
                                []).append(i)
    keep_coords, keep_vals, keep_covs, keep_batch, keep_ids, counts = \
        [], [], [], [], [], []
    for key, grp in key_to_group.items():
        if len(grp) == 1:
            i = grp[0]
            keep_coords.append(coords[i])
            keep_vals.append(values[i])
            keep_covs.append(covs[i])
            keep_batch.append(batches[i])
            keep_ids.append(ids[i])
            counts.append(1)
            continue
        vals = values[grp]
        merged_val = float(np.mean(vals))
        bset = sorted(set(str(b) for b in batches[grp]))
        if len(bset) > 1:
            note = f"跨批次 {bset} 的重复坐标被合并，批次取 {bset[0]}"
        else:
            note = "同批次重复"
        report.merges.append({
            "type": "exact_duplicate",
            "ids": [str(ids[i]) for i in grp],
            "at": [float(key[0]), float(key[1])],
            "values": [float(v) for v in vals],
            "merged_value": merged_val,
            "count": len(grp),
            "note": note + f"；块金缩小为 1/{len(grp)}",
        })
        keep_coords.append(np.array(key))
        keep_vals.append(merged_val)
        keep_covs.append(np.mean(covs[grp], axis=0))
        keep_batch.append(batches[grp[0]])
        keep_ids.append("+".join(str(ids[i]) for i in grp))
        counts.append(len(grp))
    return (np.array(keep_coords), np.array(keep_vals, dtype=float),
            np.array(keep_covs), np.array(keep_batch, dtype=object),
            np.array(keep_ids, dtype=object),
            np.array(counts, dtype=int))


def handle_near_duplicates(coords, values, covs, batches, ids, counts,
                           eps, strategy, report, seed=7):
    """处理距离 < eps 的近重合点对。strategy: 'merge' | 'jitter'。"""
    n = len(values)
    if n < 2 or eps <= 0:
        return coords, values, covs, batches, ids, counts
    d = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((d**2).sum(-1))
    iu = np.triu_indices(n, k=1)
    close = np.where(D[iu] < eps)[0]
    if len(close) == 0:
        return coords, values, covs, batches, ids, counts

    if strategy == "jitter":
        rng = np.random.default_rng(seed)
        new_coords = coords.copy()
        moved = []
        for k in close:
            i, j = iu[0][k], iu[1][k]
            shift = rng.uniform(-eps, eps, size=2)
            new_coords[j] = coords[j] + shift
            moved.append({
                "type": "near_jitter",
                "ids": [str(ids[i]), str(ids[j])],
                "distance": float(D[iu[0][k], iu[1][k]]),
                "action": f"测点 {ids[j]} 坐标微扰 {np.round(shift, 6).tolist()}",
            })
        report.near.extend(moved)
        return new_coords, values, covs, batches, ids, counts

    # merge 策略：把相互接近的连通分量合并到质心
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in close:
        i, j = int(iu[0][k]), int(iu[1][k])
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    kc, kv, kcov, kb, ki, kcnt = [], [], [], [], [], []
    for root, grp in groups.items():
        if len(grp) == 1:
            i = grp[0]
            kc.append(coords[i]); kv.append(values[i]); kcov.append(covs[i])
            kb.append(batches[i]); ki.append(ids[i]); kcnt.append(counts[i])
            continue
        w = counts[grp].astype(float)
        m_val = float(np.average(values[grp], weights=w))
        m_coord = np.average(coords[grp], axis=0, weights=w)
        m_cnt = int(counts[grp].sum())
        report.near.append({
            "type": "near_merge",
            "ids": [str(ids[i]) for i in grp],
            "min_distance": float(min(D[iu[0][k], iu[1][k]] for k in close
                                      if find(int(iu[0][k])) == root)),
            "merged_at": [float(m_coord[0]), float(m_coord[1])],
            "values": [float(values[i]) for i in grp],
            "merged_value": m_val,
            "count": m_cnt,
            "action": "合并到质心，块金按累计测量次数缩小",
        })
        kc.append(m_coord); kv.append(m_val)
        kcov.append(np.average(covs[grp], axis=0, weights=w))
        kb.append(batches[grp[0]])
        ki.append("+".join(str(ids[i]) for i in grp))
        kcnt.append(m_cnt)
    return (np.array(kc), np.array(kv, dtype=float), np.array(kcov),
            np.array(kb, dtype=object), np.array(ki, dtype=object),
            np.array(kcnt, dtype=int))


def ensure_well_conditioned(C, sill, report, cond_threshold=1e10,
                            max_tries=8):
    """检查病态；必要时逐级加对角 jitter。返回 (C_fixed, jitter)。"""
    n = C.shape[0]
    scale = max(float(sill), 1e-12)

    def chol_ok(M):
        try:
            np.linalg.cholesky(M)
            return True
        except np.linalg.LinAlgError:
            return False

    def cond(M):
        try:
            return float(np.linalg.cond(M))
        except np.linalg.LinAlgError:
            return np.inf

    C_fix = np.array(C, dtype=float, copy=True)
    cond_before = cond(C_fix)
    jitter = 0.0
    ok = chol_ok(C_fix) and cond_before <= cond_threshold
    tries = 0
    while not ok and tries < max_tries:
        jitter = max(jitter * 10, scale * 1e-10)
        C_fix = C + jitter * np.eye(n)
        ok = chol_ok(C_fix) and cond(C_fix) <= cond_threshold
        tries += 1
    if not chol_ok(C_fix):
        raise np.linalg.LinAlgError("加噪后协方差矩阵仍非正定")
    if jitter > 0:
        report.conditioning = {
            "jitter": float(jitter),
            "relative_to_sill": float(jitter / scale),
            "condition_before": cond_before if np.isfinite(cond_before) else "inf",
            "condition_after": cond(C_fix),
            "note": "协方差矩阵病态，对角加噪稳定；等效于增大块金",
        }
    return C_fix, jitter
