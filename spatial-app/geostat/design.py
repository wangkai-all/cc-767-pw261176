"""自适应加点设计：兼顾现有采样相关性与费用。

在位置 s 新增一次测量后，目标格 t 的预测方差精确下降

    Δvar(t|s) = c_resid(s,t)^2 / (var_resid(s) + nugget)

其中 c_resid 是扣除当前数据与趋势不确定性后的残差协方差
（克里金方差不依赖观测值，因此无需知道 s 处的取值即可评分）。

费用:  cost(s) = base_cost + travel_cost * 到最近现有测点的（测地）距离
评分:  score(s) = Σ_t Δvar(t|s) / cost(s)
贪心序贯选取，每选一个就把它作为伪测点并入系统再评下一个。
"""
from __future__ import annotations

import numpy as np

from . import covariance as cov


def integrated_variance_reduction(krig, params, D_cand_train, F_cand,
                                  D_cand_cells, cells_cache, nugget):
    """对每个候选点计算总方差下降与自身残差方差。

    krig: 当前 UniversalKriging；D_cand_train: (m,n)；F_cand: (m,p)；
    D_cand_cells: (m, n_cells)；cells_cache: predict_cells 的缓存；
    返回 (dvar (m,), var_ss (m,))。
    """
    sill = params["sill"] if isinstance(params, dict) else params.sill
    model = params["model"] if isinstance(params, dict) else params.model
    rng = params["range"] if isinstance(params, dict) else params.range

    c_s = sill * cov.corr(D_cand_train, model, rng)      # (m, n)
    Ci_cs = krig.solve(c_s.T)                            # (n, m)
    r_s = F_cand.T - krig.F.T @ Ci_cs                    # (p, m)
    var_ss = sill - np.einsum("mn,nm->m", c_s, Ci_cs) \
        + np.einsum("pm,pq,qm->m", r_s, krig.Ginv, r_s)
    var_ss = np.maximum(var_ss, 0.0)

    c_t, Cic_t, R_t = cells_cache
    c_st = sill * cov.corr(D_cand_cells, model, rng)     # (m, n_cells)
    c_resid = c_st - Ci_cs.T @ c_t + r_s.T @ krig.Ginv @ R_t
    dvar = (c_resid**2).sum(axis=1) / (var_ss + nugget)
    return dvar, var_ss


def candidate_cost(D_cand_train, base_cost, travel_cost):
    """费用 = 基础费 + 行进费 × 到最近现有测点的距离。"""
    dmin = D_cand_train.min(axis=1)
    dmin = np.where(np.isfinite(dmin), dmin, 10 * np.nanmax(dmin[np.isfinite(dmin)])
                    if np.isfinite(dmin).any() else 1.0)
    return base_cost + travel_cost * dmin
