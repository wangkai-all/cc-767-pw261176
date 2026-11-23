"""屏障（遮挡边界）行为核对：测地距离、相关阻断、无支持标记。"""
import tempfile

import numpy as np
import pytest

from geostat.dataset import Dataset
from geostat.geom import BarrierSet, point_in_polygon
from geostat.model import SpatialModel

# 竖直薄墙：x∈[49,51], y∈[0,80]，上方 y>80 留有缺口
WALL = [[49, 0], [51, 0], [51, 80], [49, 80]]
# 全高墙：把区域完全切成两半
FULL_WALL = [[49, -5], [51, -5], [51, 105], [49, 105]]


def test_geodesic_distance_detours_around_wall():
    bs = BarrierSet([WALL])
    pts = np.array([[44.0, 40.0], [54.0, 40.0]])
    D = bs.pairwise(pts)
    euclid = 10.0
    assert D[0, 1] > 4 * euclid, "穿墙对的测地距离应显著大于欧氏距离"
    # 不绕路的对仍等于欧氏距离
    pts2 = np.array([[40.0, 40.0], [44.0, 40.0]])
    D2 = bs.pairwise(pts2)
    assert D2[0, 1] == pytest.approx(4.0)


def test_full_wall_isolates():
    """两端接到区域边界的墙把区域切开：两侧点对完全不相关。"""
    bs = BarrierSet([FULL_WALL], bounds=(0, 0, 100, 100))
    pts = np.array([[30.0, 50.0], [70.0, 50.0]])
    D = bs.pairwise(pts)
    assert np.isinf(D[0, 1]), "被全高墙隔开的点对距离应为 inf（不相关）"
    # 不设置区域边界时，有限墙总能绕行（距离有限但很大）
    bs_open = BarrierSet([FULL_WALL])
    D_open = bs_open.pairwise(pts)
    assert np.isfinite(D_open[0, 1]) and D_open[0, 1] > 100


def _clustered_two_sides(seed=5):
    """墙两侧各一簇点。"""
    rng = np.random.default_rng(seed)
    left = np.column_stack([rng.normal(44, 2.5, 25), rng.normal(40, 6, 25)])
    right = np.column_stack([rng.normal(54, 2.5, 25), rng.normal(40, 6, 25)])
    coords = np.vstack([left, right])
    # 两侧独立的信号（墙阻断），各加小噪声
    y = np.concatenate([10 + rng.normal(0, 0.3, 25),
                        0 + rng.normal(0, 0.3, 25)])
    return coords, y


def test_barrier_blocks_correlation():
    """墙存在时，对侧点对预测权重应近似为 0；去掉墙则不为 0。"""
    coords, y = _clustered_two_sides()
    n = len(y)
    ds = Dataset(np.array([f"p{i}" for i in range(n)], dtype=object),
                 coords, y, np.array([""] * n, dtype=object),
                 np.zeros((n, 0)), [])
    cfg = {"model": "exponential", "use_xy_trend": False,
           "use_covariates": False}
    m_wall = SpatialModel(ds, barriers=[WALL], config=dict(cfg),
                          versions_dir=tempfile.mkdtemp())
    m_wall.fit()
    # 预测点紧贴墙左侧
    w = m_wall.weights_at(46.0, 40.0, top=50)
    right_weight = sum(abs(t["w"]) for t in w["weights"] if t["x"] > 51)
    left_weight = sum(abs(t["w"]) for t in w["weights"] if t["x"] < 49)
    assert right_weight < 0.02 * max(left_weight, 1e-9), \
        f"墙对侧权重 {right_weight} 应近似为 0（左侧 {left_weight}）"

    m_open = SpatialModel(ds, barriers=[WALL],
                          config=dict(cfg, use_barriers=False),
                          versions_dir=tempfile.mkdtemp())
    m_open.fit()
    w2 = m_open.weights_at(46.0, 40.0, top=50)
    right_weight2 = sum(abs(t["w"]) for t in w2["weights"] if t["x"] > 51)
    assert right_weight2 > 3 * right_weight, \
        "关闭屏障后对侧权重应明显增大（屏障确实在阻断相关）"


def test_unsupported_marking_across_wall():
    """只在墙左侧采样：墙右侧（绕路远）应被标记为无支持；
    关闭屏障后同一位置变为有支持。"""
    from geostat.geom import BarrierSet as BS
    from geostat.simulate import simulate_field
    rng = np.random.default_rng(2)
    left = np.column_stack([rng.normal(40, 5, 40), rng.normal(40, 10, 40)])
    # 有真实空间结构的数据（变程 ~25），使参数估计稳定
    Dl = BS(None).pairwise(left)
    F1 = np.ones((40, 1))
    y, _ = simulate_field(F1, [5.0], "exponential", 2.0, 25.0, 0.1,
                          D=Dl, seed=2)
    ds = Dataset(np.array([f"p{i}" for i in range(40)], dtype=object),
                 left, y, np.array([""] * 40, dtype=object),
                 np.zeros((40, 0)), [])
    cfg = {"model": "exponential", "use_xy_trend": False,
           "use_covariates": False}
    m = SpatialModel(ds, barriers=[WALL], config=dict(cfg),
                     versions_dir=tempfile.mkdtemp())
    m.fit()
    g = m.predict_grid(nx=50, ny=50, save_version=False)
    gx, gy = g["gx"], g["gy"]
    jx = int(np.argmin(np.abs(gx - 60)))   # 墙右侧 (60,40)
    jy = int(np.argmin(np.abs(gy - 40)))
    assert g["unsupported"][jy, jx], "墙后远处格点应标记为无支持"

    m2 = SpatialModel(ds, barriers=[WALL], config=dict(cfg,
                                                       use_barriers=False),
                      versions_dir=tempfile.mkdtemp())
    m2.fit()
    g2 = m2.predict_grid(nx=50, ny=50, save_version=False)
    assert not g2["unsupported"][jy, jx], "无屏障时该格点应有支持"


def test_blocked_cells_masked():
    """屏障多边形内部格点应被标记为阻断。"""
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 100, size=(40, 2))
    y = rng.normal(0, 1, 40)
    ds = Dataset(np.array([f"p{i}" for i in range(40)], dtype=object),
                 coords, y, np.array([""] * 40, dtype=object),
                 np.zeros((40, 0)), [])
    m = SpatialModel(ds, barriers=[WALL],
                     config={"use_xy_trend": False, "use_covariates": False},
                     versions_dir=tempfile.mkdtemp())
    m.fit()
    g = m.predict_grid(nx=50, ny=50, save_version=False)
    assert g["blocked"].sum() > 0
    # 阻断格一定在墙内
    GX, GY = np.meshgrid(g["gx"], g["gy"])
    for jy, jx in zip(*np.where(g["blocked"])):
        assert point_in_polygon((GX[jy, jx], GY[jy, jx]),
                                np.asarray(WALL, dtype=float))
