"""相关模型与协方差装配。

所有模型统一用"实用变程" a 参数化：相关度在 h = a 处降到约 0.05
（球状模型例外，a 恰为变程）。协方差结构：

    C(h) = sill * rho(h; a)        空间相关部分（sill = 偏基台值）
         + nugget_i * 1[h = 0]     测量噪声 / 微结构（逐点，合并测点会减小）

距离为 inf（被屏障完全隔离）时相关为 0。
"""
from __future__ import annotations

import numpy as np

MODELS = ("exponential", "gaussian", "matern32", "spherical")

# Matérn-3/2 的相关度在 h ≈ 4.7439 * l 处降到 0.05
_M32_PRACTICAL = 4.7439


def corr(h: np.ndarray, model: str, range_: float) -> np.ndarray:
    """相关函数 rho(h)。h 可为任意形状数组；inf 距离 -> 0。"""
    h = np.asarray(h, dtype=float)
    h = np.where(np.isfinite(h), h, 1e30)  # 屏障隔离 -> 相关 0
    a = max(float(range_), 1e-12)
    if model == "exponential":
        r = np.exp(-3.0 * h / a)
    elif model == "gaussian":
        r = np.exp(-3.0 * (h / a) ** 2)
    elif model == "matern32":
        u = np.sqrt(3.0) * h / (a / _M32_PRACTICAL)
        r = (1.0 + u) * np.exp(-u)
    elif model == "spherical":
        u = h / a
        r = np.where(u < 1.0, 1.0 - 1.5 * u + 0.5 * u**3, 0.0)
    else:
        raise ValueError(f"未知相关模型: {model!r}，可选 {MODELS}")
    return r


def covariance(D: np.ndarray, model: str, sill: float, range_: float) -> np.ndarray:
    """由距离矩阵给出（互）协方差，不含块金。对角处为 sill。"""
    return float(sill) * corr(D, model, range_)


def add_diagonal_noise(C: np.ndarray, noise: np.ndarray | float) -> np.ndarray:
    """对角加噪（块金 / 稳定化 jitter），返回新矩阵。"""
    C = np.array(C, dtype=float, copy=True)
    idx = np.diag_indices_from(C)
    C[idx] += noise
    return C
