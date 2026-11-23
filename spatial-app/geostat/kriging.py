"""泛克里金（带趋势项的空间插值）。

模型:  y = F beta + eps,  Cov(eps) = C
预测:  y0_hat = c0' C^-1 y + (f0 - F' C^-1 c0)' beta_hat
方差:  c00 - c0' C^-1 c0 + r' (F' C^-1 F)^-1 r,  r = f0 - F' C^-1 c0

- target='field'       预测潜场（c00 = sill）
- target='measurement' 预测一次新测量（c00 = sill + 新测点块金）

权重 w = C^-1 c0 + C^-1 F G^-1 r 一并返回，供界面"点选预测格追到邻点权重"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import covariance as cov
from .preprocessing import ensure_well_conditioned, PreprocessReport


@dataclass
class KrigingParams:
    model: str
    sill: float          # 偏基台值（空间相关部分方差）
    range: float         # 实用变程
    nugget: float        # 单次测量的块金方差

    def to_dict(self):
        return {"model": self.model, "sill": self.sill,
                "range": self.range, "nugget": self.nugget,
                "total_var": self.sill + self.nugget}


class UniversalKriging:
    """在固定参数下拟合泛克里金系统并对新位置预测。"""

    def __init__(self, D: np.ndarray, F: np.ndarray, y: np.ndarray,
                 params: KrigingParams, counts: np.ndarray | None = None,
                 report: PreprocessReport | None = None):
        """D: (n,n) 距离矩阵；F: (n,p) 趋势设计矩阵；y: (n,) 观测。
        counts: 每个测点合并的测量次数（块金 = nugget/counts）。"""
        self.params = params
        self.F = np.asarray(F, dtype=float)
        self.y = np.asarray(y, dtype=float)
        n = len(self.y)
        counts = np.ones(n) if counts is None else np.asarray(counts, float)

        C = cov.covariance(D, params.model, params.sill, params.range)
        C = cov.add_diagonal_noise(C, params.nugget / counts)
        if report is not None:
            C, self.jitter = ensure_well_conditioned(C, params.sill, report)
        else:
            C, self.jitter = ensure_well_conditioned(
                C, params.sill, PreprocessReport())
        self.C = C
        self._chol = cho_factor(C, lower=True)

        CiF = cho_solve(self._chol, self.F)
        G = self.F.T @ CiF
        # 趋势列可能秩亏：用 pinv 兜底
        self._Ginv = np.linalg.pinv(G)
        self._CiF = CiF
        self.beta = self._Ginv @ (CiF.T @ self.y)
        self.resid = self.y - self.F @ self.beta

    # --------------------------------------------------------------

    def _cross(self, D0: np.ndarray):
        """新位置交叉协方差与趋势残差。D0: (n, m)。"""
        c0 = cov.covariance(D0, self.params.model,
                            self.params.sill, self.params.range)
        Cic0 = cho_solve(self._chol, c0)
        return c0, Cic0

    def trend_residual(self, D0: np.ndarray, F0: np.ndarray):
        """返回 (c0, Cic0, R)：交叉协方差、C^-1 c0、趋势残差 R = f0 - F'C^-1 c0。

        供 design.py 计算残差协方差 c_resid(s,t) = c(s,t) - c_s'C^-1 c_t
        + r_s' Ginv r_t。
        """
        F0 = np.atleast_2d(np.asarray(F0, dtype=float))
        c0, Cic0 = self._cross(D0)
        R = F0.T - self.F.T @ Cic0
        return c0, Cic0, R

    @property
    def Ginv(self):
        return self._Ginv

    def solve(self, B):
        """解 C X = B（用已分解的 Cholesky）。"""
        return cho_solve(self._chol, B)

    def predict(self, D0: np.ndarray, F0: np.ndarray,
                target: str = "field", nugget_new: float | None = None,
                return_weights: bool = False):
        """D0: (n,m) 训练点-目标距离；F0: (m,p) 目标趋势行。

        返回 dict(mean, var, sd, weights?)，weights 形状 (n, m)。
        """
        F0 = np.atleast_2d(np.asarray(F0, dtype=float))
        m = F0.shape[0]
        c0, Cic0 = self._cross(D0)
        R = F0.T - self.F.T @ Cic0          # (p, m)
        mean = Cic0.T @ self.y + R.T @ self.beta
        base = self.params.sill if target == "field" \
            else self.params.sill + (self.params.nugget
                                     if nugget_new is None else nugget_new)
        var = base - np.einsum("nm,nm->m", c0, Cic0) \
            + np.einsum("pm,pq,qm->m", R, self._Ginv, R)
        n_neg = int((var < 0).sum())
        var = np.maximum(var, 0.0)
        out = {"mean": mean, "var": var, "sd": np.sqrt(var),
               "n_negative_var_clipped": n_neg}
        if return_weights:
            W = Cic0 + self._CiF @ (self._Ginv @ R)
            out["weights"] = W
        return out


def trend_matrix(coords: np.ndarray, covs: np.ndarray | None,
                 use_xy: bool, cov_means=None, cov_stds=None,
                 domain_scale: float | None = None):
    """构建趋势设计矩阵 [1, x, y, 协变量...]（标准化）。

    返回 (F, transform)，transform 供预测时复用同一标准化。
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if domain_scale is None:
        domain_scale = float(max(np.ptp(coords[:, 0]),
                                 np.ptp(coords[:, 1]), 1.0))
    cols = [np.ones(n)]
    names = ["intercept"]
    if use_xy:
        cols += [coords[:, 0] / domain_scale, coords[:, 1] / domain_scale]
        names += ["x", "y"]
    if covs is not None and np.asarray(covs).size:
        covs = np.asarray(covs, dtype=float)
        if cov_means is None:
            cov_means = covs.mean(0)
            cov_stds = np.where(covs.std(0) > 0, covs.std(0), 1.0)
        Z = (covs - cov_means) / cov_stds
        for j in range(Z.shape[1]):
            cols.append(Z[:, j])
            names.append(f"cov{j}")
    else:
        cov_means = cov_stds = None
    F = np.column_stack(cols)
    transform = {"use_xy": use_xy, "domain_scale": domain_scale,
                 "cov_means": None if cov_means is None else list(cov_means),
                 "cov_stds": None if cov_stds is None else list(cov_stds),
                 "names": names}
    return F, transform


def apply_trend_transform(coords, covs, transform):
    """用已存的标准化参数构建趋势矩阵。"""
    return trend_matrix(coords, covs, transform["use_xy"],
                        None if transform["cov_means"] is None
                        else np.array(transform["cov_means"]),
                        None if transform["cov_stds"] is None
                        else np.array(transform["cov_stds"]),
                        transform["domain_scale"])[0]


def drop_collinear(F, names, report: PreprocessReport | None = None,
                   tol=1e-10):
    """剔除共线趋势列（QR 秩检验），返回 (F_kept, kept_names)。"""
    keep = []
    R_rank = 0
    A = np.empty((F.shape[0], 0))
    for j in range(F.shape[1]):
        B = np.column_stack([A, F[:, j]])
        r = np.linalg.matrix_rank(B, tol=tol)
        if r > R_rank:
            keep.append(j)
            A = B
            R_rank = r
        elif report is not None:
            report.trend.setdefault("dropped_columns", []).append(names[j])
    return F[:, keep], [names[j] for j in keep]
