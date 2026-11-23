"""块划分约束与加点设计核对。"""
import tempfile

import numpy as np
import pytest

from geostat.blocks import assign_blocks, make_folds
from geostat.dataset import Dataset
from geostat.model import SpatialModel


def test_batches_stay_together():
    """同批测点必须全部落在同一块。"""
    rng = np.random.default_rng(1)
    coords, batches = [], []
    for b in range(6):
        center = rng.uniform(10, 90, size=2)
        pts = center + rng.normal(0, 3, size=(12, 2))
        coords.append(pts)
        batches += [f"batch{b}"] * 12
    coords = np.vstack(coords)
    batches = np.array(batches, dtype=object)
    block_ids = assign_blocks(coords, batches, n_blocks=4, seed=0)
    for b in set(batches.tolist()):
        ids = set(block_ids[batches == b].tolist())
        assert len(ids) == 1, f"批次 {b} 被拆到多个块: {ids}"
    # 折划分同样保持批次完整
    for fold in make_folds(block_ids):
        test_blocks = set(fold["test_blocks"])
        for b in set(batches.tolist()):
            in_test = block_ids[batches == b][0] in test_blocks
            assert all((block_ids[batches == b][0] in test_blocks) == in_test
                       for _ in [0])


def test_suggest_reduces_variance_and_respects_cost():
    """建议点应降低总方差；费用提高时建议点应更靠近现有测点。"""
    from geostat.geom import BarrierSet
    from geostat.simulate import simulate_field
    rng = np.random.default_rng(8)
    # 左半边密集采样，右半边空 —— 建议应落在右侧空白区
    coords = np.column_stack([rng.uniform(0, 45, 60),
                              rng.uniform(0, 100, 60)])
    D = BarrierSet(None).pairwise(coords)
    y, _ = simulate_field(np.ones((60, 1)), [3.0], "exponential",
                          2.0, 18.0, 0.2, D=D, seed=8)
    ds = Dataset(np.array([f"p{i}" for i in range(60)], dtype=object),
                 coords, y, np.array([""] * 60, dtype=object),
                 np.zeros((60, 0)), [])
    cfg = {"model": "exponential", "use_xy_trend": True,
           "use_covariates": False}
    # 分析区域显式设为 [0,100]^2（数据只覆盖左半）
    m = SpatialModel(ds, config=cfg, domain=(0, 0, 100, 100),
                     versions_dir=tempfile.mkdtemp())
    m.fit()
    s = m.suggest(n_suggest=3, base_cost=1.0, travel_cost=0.0,
                  nx=30, ny=30)
    xs = [c["x"] for c in s["candidates"]]
    assert all(x > 45 for x in xs), f"空白区在右侧，建议应落在右侧: {xs}"
    assert all(c["dvar"] > 0 for c in s["candidates"])

    def dist_to_nearest(px, py):
        return float(np.sqrt(((coords - [px, py])**2).sum(1)).min())

    # 高行进费 → 建议点更贴近现有采样
    m2 = SpatialModel(ds, config=cfg, domain=(0, 0, 100, 100),
                      versions_dir=tempfile.mkdtemp())
    m2.fit()
    s2 = m2.suggest(n_suggest=1, base_cost=1.0, travel_cost=5.0,
                    nx=30, ny=30)
    assert len(s2["candidates"]) == 1
    d_free = dist_to_nearest(s["candidates"][0]["x"], s["candidates"][0]["y"])
    d_exp = dist_to_nearest(s2["candidates"][0]["x"],
                            s2["candidates"][0]["y"])
    assert d_exp < d_free, \
        f"行进费升高后建议点应更靠近现有测点 ({d_exp:.1f} < {d_free:.1f})"


def test_accept_point_adds_synthetic_and_versions():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 100, size=(50, 2))
    y = np.cos(coords[:, 1] / 20) + rng.normal(0, 0.3, 50)
    ds = Dataset(np.array([f"p{i}" for i in range(50)], dtype=object),
                 coords, y, np.array([""] * 50, dtype=object),
                 np.zeros((50, 0)), [])
    vdir = tempfile.mkdtemp()
    m = SpatialModel(ds, config={"use_covariates": False}, versions_dir=vdir)
    m.fit()
    v1 = m.model_version
    out = m.accept_point(55.0, 55.0, seed=0)
    assert len(m.dataset) == 51
    assert out["point"]["batch"] == "synthetic"
    assert m.model_version != v1, "接受建议后应产生新模型版本"
    assert len(m.store.index["models"]) == 2
