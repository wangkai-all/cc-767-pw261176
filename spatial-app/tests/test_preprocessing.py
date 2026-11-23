"""预处理核对：精确重复、近重合、数值病态分别处理且留有报告。"""
import numpy as np
import pytest

from geostat.kriging import KrigingParams, UniversalKriging
from geostat.preprocessing import (PreprocessReport, ensure_well_conditioned,
                                   handle_near_duplicates,
                                   merge_exact_duplicates)


def _arrays(coords, values=None):
    n = len(coords)
    coords = np.asarray(coords, dtype=float)
    values = np.arange(n, dtype=float) if values is None \
        else np.asarray(values, dtype=float)
    covs = np.zeros((n, 0))
    batches = np.array(["b"] * n, dtype=object)
    ids = np.array([f"p{i}" for i in range(n)], dtype=object)
    return coords, values, covs, batches, ids


def test_exact_duplicates_merged():
    coords, values, covs, batches, ids = _arrays(
        [[0, 0], [1, 1], [0, 0], [2, 2], [0, 0]], [1, 2, 3, 4, 5])
    rep = PreprocessReport()
    out = merge_exact_duplicates(coords, values, covs, batches, ids, rep)
    c, v, _, _, _, counts = out
    assert len(v) == 3                      # 5 点 -> 3 点
    i0 = int(np.where((c == [0.0, 0.0]).all(1))[0][0])
    assert v[i0] == pytest.approx(np.mean([1, 3, 5]))
    assert counts[i0] == 3
    assert len(rep.merges) == 1
    assert rep.merges[0]["count"] == 3
    assert "1/3" in rep.merges[0]["note"]   # 报告说明了块金变化


def test_near_duplicates_merge_vs_jitter():
    coords, values, covs, batches, ids = _arrays(
        [[0, 0], [1e-4, 0], [5, 5]], [1.0, 3.0, 9.0])
    counts = np.ones(3, dtype=int)

    rep_m = PreprocessReport()
    out = handle_near_duplicates(coords, values, covs, batches, ids,
                                 counts, eps=1e-3, strategy="merge",
                                 report=rep_m)
    assert len(out[1]) == 2                 # 前两点合并
    assert out[1][0] == pytest.approx(2.0)  # 均值
    assert out[5][0] == 2                   # 累计测量次数
    assert rep_m.near[0]["type"] == "near_merge"

    rep_j = PreprocessReport()
    out_j = handle_near_duplicates(coords, values, covs, batches, ids,
                                   counts, eps=1e-3, strategy="jitter",
                                   report=rep_j)
    assert len(out_j[1]) == 3               # 不合并
    moved = np.abs(out_j[0][1] - coords[1]).sum()
    assert moved > 0
    assert rep_j.near[0]["type"] == "near_jitter"
    assert "微扰" in rep_j.near[0]["action"]


def test_ill_conditioning_gets_jitter_and_report():
    """高斯相关 + 极近点 → 病态；应自动加噪并报告改变了什么。"""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 50, size=(30, 2))
    coords = np.vstack([base, base[0] + 1e-9])   # 几乎重合
    n = len(coords)
    d = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((d**2).sum(-1))
    F = np.column_stack([np.ones(n), coords[:, 0] / 50, coords[:, 1] / 50])
    y = rng.normal(0, 1, n)
    rep = PreprocessReport()
    # 高斯模型、长变程、零块金 → 典型病态
    krig = UniversalKriging(D, F, y, KrigingParams("gaussian", 2.0, 80.0,
                                                   0.0),
                            report=rep)
    assert rep.conditioning.get("jitter", 0) > 0
    assert "condition_after" in rep.conditioning
    # 加噪后系统可用（D0 形状为 (n_train, m_target)）
    out = krig.predict(D[:, :1], F[:1])
    assert np.isfinite(out["mean"][0])


def test_well_conditioned_matrix_untouched():
    D = np.array([[0.0, 5.0], [5.0, 0.0]])
    from geostat import covariance as cov
    C = cov.covariance(D, "exponential", 2.0, 10.0)
    C = cov.add_diagonal_noise(C, 0.5)
    rep = PreprocessReport()
    _, jitter = ensure_well_conditioned(C, 2.0, rep)
    assert jitter == 0.0
    assert not rep.conditioning
