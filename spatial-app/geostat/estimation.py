"""协方差参数估计：剖面 REML，只用传入的（训练）数据。

对固定的 (变程 a, 块金比 rho)：
    K = (1-rho) R(a) + rho * diag(1/counts)     （相关矩阵 + 相对块金）
    beta_hat = (F'K^-1 F)^-1 F'K^-1 y
    phi_hat  = r'K^-1 r / (n - p)                （总方差）
    REML 对数似然 = -0.5 [(n-p) log phi_hat + log|K| + log|F'K^-1 F|] + const

先粗网格搜索，再 Nelder-Mead 细化。返回 sill = (1-rho) phi, nugget = rho phi。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from . import covariance as cov


def _reml_loglik(log_a, logit_rho, D, F, y, model, counts):
    a = np.exp(log_a)
    rho = 1.0 / (1.0 + np.exp(-logit_rho))
    n, p = F.shape
    if n - p < 3:
        return -np.inf, None
    R = cov.corr(D, model, a)
    K = (1.0 - rho) * R
    K[np.diag_indices_from(K)] += rho / counts
    K += 1e-10 * np.eye(n)
    try:
        chol = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf, None
    KiF = cho_solve(chol, F)
    G = F.T @ KiF
    sign, logdetK = np.linalg.slogdet(K)
    if sign <= 0:
        return -np.inf, None
    try:
        Ginv = np.linalg.pinv(G)
        signG, logdetG = np.linalg.slogdet(G)
        if signG <= 0:
            logdetG = 0.0
    except np.linalg.LinAlgError:
        return -np.inf, None
    beta = Ginv @ (KiF.T @ y)
    r = y - F @ beta
    Kir = cho_solve(chol, r)
    quad = float(r @ Kir)
    if quad <= 0:
        return -np.inf, None
    phi = quad / (n - p)
    ll = -0.5 * ((n - p) * np.log(phi) + logdetK + logdetG)
    return ll, (a, rho, phi)


def estimate_params(D, F, y, model, counts=None, domain_diag=None,
                    refine=True):
    """估计 (sill, range, nugget)。返回 (params_dict, diagnostics)。"""
    D = np.asarray(D, dtype=float)
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    counts = np.ones(n) if counts is None else np.asarray(counts, float)
    if domain_diag is None:
        finite = D[np.isfinite(D)]
        domain_diag = float(finite.max()) if finite.size else 1.0
        domain_diag = max(domain_diag, 1e-6)

    # 粗网格
    a_grid = np.geomspace(0.03 * domain_diag, 1.5 * domain_diag, 9)
    rho_grid = np.array([0.001, 0.01, 0.05, 0.15, 0.3, 0.5])
    best = (-np.inf, None)
    for a in a_grid:
        for rho in rho_grid:
            ll, res = _reml_loglik(np.log(a), np.log(rho / (1 - rho)),
                                   D, F, y, model, counts)
            if ll > best[0]:
                best = (ll, res)
    if best[1] is None:
        # 数据几乎无结构：退回弱相关默认
        a0, rho0, phi0 = 0.3 * domain_diag, 0.1, float(np.var(y) + 1e-9)
    else:
        a0, rho0, phi0 = best[1]

    evals = {"grid_best": {"range": a0, "nugget_ratio": rho0}}
    if refine:
        # 有界细化：变程不跑出合理窗口，避免退化数据下发散
        lo_a, hi_a = np.log(0.02 * domain_diag), np.log(2.0 * domain_diag)
        lo_r, hi_r = np.log(1e-4 / (1 - 1e-4)), np.log(0.9 / 0.1)

        def neg(x):
            ll, _ = _reml_loglik(x[0], x[1], D, F, y, model, counts)
            return -ll if np.isfinite(ll) else 1e12

        x0 = np.array([np.clip(np.log(a0), lo_a, hi_a),
                       np.clip(np.log(rho0 / (1 - rho0)), lo_r, hi_r)])
        res = minimize(neg, x0, method="L-BFGS-B",
                       bounds=[(lo_a, hi_a), (lo_r, hi_r)],
                       options={"maxiter": 200})
        ll, out = _reml_loglik(res.x[0], res.x[1], D, F, y, model, counts)
        if out is not None and np.isfinite(ll) and ll >= best[0] - 1e-6:
            a0, rho0, phi0 = out
        evals["refine_success"] = bool(res.success)
        evals["range_bounds"] = [float(np.exp(lo_a)), float(np.exp(hi_a))]

    sill = (1.0 - rho0) * phi0
    nugget = rho0 * phi0
    params = {"model": model, "sill": float(sill), "range": float(a0),
              "nugget": float(nugget), "total_var": float(sill + nugget),
              "nugget_ratio": float(rho0)}
    return params, evals
