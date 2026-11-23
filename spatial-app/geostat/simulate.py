"""已知随机场模拟：用于测试核对（方差、覆盖率）与合成测量。"""
from __future__ import annotations

import numpy as np

from . import covariance as cov


def simulate_field(F, beta, model, sill, range_, nugget, D=None,
                   counts=None, seed=None):
    """y = F beta + GP(0, sill*rho) + N(0, nugget/counts)。

    D 为距离矩阵（可含屏障测地距离）；None 时由 F 行数无法推断，
    因此必须显式传入。
    """
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    C = sill * cov.corr(D, model, range_)
    C += 1e-10 * np.eye(n)
    L = np.linalg.cholesky(C)
    field = L @ rng.standard_normal(n)
    noise_scale = np.sqrt(nugget / (counts if counts is not None else 1.0))
    y = F @ np.asarray(beta, dtype=float) + field \
        + noise_scale * rng.standard_normal(n)
    return y, field


def conditional_draw(mean, var, nugget, rng):
    """在给定当前数据下，于一点抽取一次新测量（潜场 + 测量噪声）。"""
    return float(mean + np.sqrt(max(var, 0.0)) * rng.standard_normal()
                 + np.sqrt(nugget) * rng.standard_normal())
