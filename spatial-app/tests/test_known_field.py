"""用已知随机场核对：参数估计、克里金方差校准、区间覆盖率，
以及块交叉验证 vs 随机拆分（后者应显著高估精度）。"""
import tempfile

import numpy as np
import pytest

from geostat.estimation import estimate_params
from geostat.geom import BarrierSet
from geostat.kriging import KrigingParams, UniversalKriging, trend_matrix
from geostat.model import SpatialModel
from geostat.dataset import Dataset
from geostat.simulate import simulate_field

# 真实参数
SILL, RANGE, NUGGET = 3.0, 20.0, 0.5
MODEL = "exponential"
BETA = [1.0, 2.0, -1.0]


def make_known_field(n=220, seed=11, clustered=True, n_test=0):
    """生成已知随机场的一次实现。返回训练/测试拆分的一致实现。"""
    rng = np.random.default_rng(seed)
    if clustered:
        centers = rng.uniform(10, 90, size=(8, 2))
        assign = rng.integers(0, 8, n)
        coords = centers[assign] + rng.normal(0, 4, size=(n, 2))
        batches = np.array([f"B{a}" for a in assign], dtype=object)
    else:
        coords = rng.uniform(0, 100, size=(n, 2))
        batches = np.array([""] * n, dtype=object)
    coords = np.clip(coords, 0, 100)
    if n_test:
        test_coords = rng.uniform(0, 100, size=(n_test, 2))
        all_coords = np.vstack([coords, test_coords])
    else:
        all_coords = coords
    D = BarrierSet(None).pairwise(all_coords)
    F_all, _ = trend_matrix(all_coords, None, True)
    y_all, gp_dev = simulate_field(F_all, BETA, MODEL, SILL, RANGE,
                                   NUGGET, D=D, seed=seed + 1)
    latent = F_all @ np.array(BETA) + gp_dev   # 潜场 = 趋势 + GP 偏差
    out = {
        "coords": coords, "y": y_all[:n], "batches": batches,
        "F": F_all[:n], "D": D[:n, :n],
    }
    if n_test:
        out.update({
            "test_coords": all_coords[n:],
            "test_field": latent[n:],          # 与 target='field' 对应
            "D_test_train": D[n:, :n],
            "F_test": F_all[n:],
        })
    return out


def test_parameter_estimation_recovers_truth():
    d = make_known_field()
    params, _ = estimate_params(d["D"], d["F"], d["y"], MODEL)
    assert 8 < params["range"] < 60, \
        f"变程估计 {params['range']:.1f} 偏离真值 {RANGE}"
    total = params["sill"] + params["nugget"]
    assert total == pytest.approx(SILL + NUGGET, rel=0.6)


def test_kriging_variance_and_coverage():
    """同一实现上，95% 预测区间对潜场的覆盖率应接近 0.95。"""
    d = make_known_field(n_test=400)
    params, _ = estimate_params(d["D"], d["F"], d["y"], MODEL)
    krig = UniversalKriging(d["D"], d["F"], d["y"], KrigingParams(
        MODEL, params["sill"], params["range"], params["nugget"]))
    out = krig.predict(d["D_test_train"].T, d["F_test"], target="field")
    z = (d["test_field"] - out["mean"]) / out["sd"]
    cov95 = float(np.mean(np.abs(z) <= 1.959963984540054))
    cov90 = float(np.mean(np.abs(z) <= 1.6448536269514722))
    assert 0.86 <= cov95 <= 0.995, f"95% 区间覆盖率 {cov95}"
    assert 0.80 <= cov90 <= 0.97, f"90% 区间覆盖率 {cov90}"
    rms_z = float(np.sqrt(np.mean(z**2)))
    assert 0.7 <= rms_z <= 1.4, f"标准化误差 RMS {rms_z}"
    assert np.all(out["var"] >= 0) and np.all(np.isfinite(out["sd"]))


def test_block_cv_vs_random_split():
    """聚集采样下，随机拆分的 RMSE 应明显小于（优于）空间块 CV ——
    即随机拆分高估精度。"""
    d = make_known_field(n=200)
    n = len(d["y"])
    covs = np.zeros((n, 0))
    ds = Dataset(np.array([f"p{i}" for i in range(n)], dtype=object),
                 d["coords"], d["y"], d["batches"], covs, [])
    m = SpatialModel(ds, config={"model": MODEL, "use_covariates": False},
                     versions_dir=tempfile.mkdtemp(prefix="geostat_ver_"))
    cv_block = m.cross_validate(scheme="block", n_folds=5, seed=3)
    cv_rand = m.cross_validate(scheme="random", n_folds=5, seed=3)
    rb, rr = cv_block["overall"]["rmse"], cv_rand["overall"]["rmse"]
    assert rb > rr * 1.03, \
        f"块CV RMSE {rb:.3f} 应明显大于随机拆分 {rr:.3f}（随机高估精度）"
    # 块 CV 覆盖率应落在合理区间
    assert 0.75 <= cv_block["overall"]["coverage95"] <= 1.0
    # 每折参数只由训练块估计，折间应有差异
    ranges = [f["params"]["range"] for f in cv_block["folds"]]
    assert len(set(round(r, 3) for r in ranges)) > 1
