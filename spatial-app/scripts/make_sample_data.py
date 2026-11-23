"""生成虚构示例数据：平面坐标 + 观测值 + 地形协变量 + 采样批次 + 屏障。

场景：100x100 区域，一道带缺口的竖直墙（x∈[48,52], y∈[5,70]）和
一个圆形湖。真实场 = 趋势(高程) + Matérn-3/2 随机场（沿测地距离，
即墙真的阻断相关）+ 测量噪声。采样为 4 个空间聚集的调查批次，
并故意放入精确重复与近重合测点以演示预处理。

用法:  python scripts/make_sample_data.py
输出:  data/points.csv, data/barriers.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geostat.geom import BarrierSet  # noqa: E402
from geostat import covariance as cov  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

WALL = [[48, 5], [52, 5], [52, 70], [48, 70]]
LAKE = [[78, 66], [86, 70], [88, 78], [82, 86], [74, 82], [72, 72]]
DOMAIN = [0, 0, 100, 100]

# 真实场参数（已知，供核对）
TRUE = {"model": "matern32", "sill": 3.5, "range": 22.0, "nugget": 0.5,
        "trend": {"intercept": 8.0, "elev": 0.06}}


def elevation(x, y):
    return 60 + 25 * np.sin(x / 28) * np.cos(y / 24) + 0.15 * (x - 50)


def moisture(x, y):
    return 0.4 + 0.3 * np.cos(x / 35) * np.sin(y / 31)


def main():
    rng = np.random.default_rng(20261124)
    # 4 个空间聚集的调查批次
    centers = [(25, 25), (25, 75), (70, 30), (60, 85)]
    coords, batches = [], []
    for bi, (cx, cy) in enumerate(centers):
        n = 26 + int(rng.integers(0, 6))
        pts = np.column_stack([
            np.clip(rng.normal(cx, 7, n), 1, 99),
            np.clip(rng.normal(cy, 7, n), 1, 99),
        ])
        coords.append(pts)
        batches += [f"survey_{chr(65 + bi)}"] * n
    # 少量散点
    pts = rng.uniform(3, 97, size=(12, 2))
    coords.append(pts)
    batches += ["survey_E"] * len(pts)
    coords = np.vstack(coords)

    # 湖内的点剔除（采样不可能落在湖里）
    bs = BarrierSet([LAKE])
    inside = np.array([np.any([p[0] >= 72 and p[0] <= 88 and p[1] >= 66
                               and p[1] <= 86]) for p in coords])
    coords = coords[~inside]
    batches = list(np.array(batches, dtype=object)[~inside])

    # 故意制造问题数据：3 组精确重复 + 2 对近重合
    dup_idx = [5, 30, 55]
    dup_rows = []
    for i in dup_idx:
        dup_rows.append((coords[i].copy(), batches[i]))
    near_rows = [(coords[10] + [0.15, -0.1], batches[10]),
                 (coords[70] + [-0.12, 0.08], batches[70])]

    all_coords = np.vstack([coords] + [np.array([r[0] for r in dup_rows + near_rows])])
    all_batches = batches + [r[1] for r in dup_rows + near_rows]

    # 真实场：趋势 + 沿测地距离的 GP（墙阻断相关）
    bs_full = BarrierSet([WALL, LAKE], bounds=(0, 0, 100, 100))
    D = bs_full.pairwise(all_coords)
    C = TRUE["sill"] * cov.corr(D, TRUE["model"], TRUE["range"]) \
        + 1e-9 * np.eye(len(all_coords))
    L = np.linalg.cholesky(C)
    gp = L @ rng.standard_normal(len(all_coords))
    elev = elevation(all_coords[:, 0], all_coords[:, 1])
    latent = TRUE["trend"]["intercept"] + TRUE["trend"]["elev"] * (elev - 60) + gp
    value = latent + np.sqrt(TRUE["nugget"]) * rng.standard_normal(len(all_coords))
    # 重复测点：重新独立测量（同点不同噪声）
    n_base = len(all_coords)
    for k, (c, b) in enumerate(dup_rows):
        all_coords = np.vstack([all_coords, c])
        all_batches.append(b)
        v = (TRUE["trend"]["intercept"] + TRUE["trend"]["elev"]
             * (elevation(c[0], c[1]) - 60)
             + np.sqrt(TRUE["sill"] + TRUE["nugget"]) * rng.standard_normal())
        value = np.append(value, v)

    moist = moisture(all_coords[:, 0], all_coords[:, 1])
    elev_all = elevation(all_coords[:, 0], all_coords[:, 1])

    os.makedirs(DATA, exist_ok=True)
    csv_path = os.path.join(DATA, "points.csv")
    with open(csv_path, "w") as f:
        f.write("id,x,y,value,batch,elev,moisture\n")
        for i in range(len(all_coords)):
            f.write(f"p{i:03d},{all_coords[i,0]:.3f},{all_coords[i,1]:.3f},"
                    f"{value[i]:.4f},{all_batches[i]},"
                    f"{elev_all[i]:.3f},{moist[i]:.4f}\n")
    with open(os.path.join(DATA, "barriers.json"), "w") as f:
        json.dump({"domain": DOMAIN,
                   "polygons": [WALL, LAKE],
                   "names": ["wall", "lake"]}, f, indent=1)
    with open(os.path.join(DATA, "truth.json"), "w") as f:
        json.dump(TRUE, f, indent=1)
    print(f"写出 {len(all_coords)} 个测点 -> {csv_path}")
    print(f"  含 {len(dup_rows)} 个精确重复、2 个近重合、4+1 个批次")
    print(f"  真实参数: {TRUE}")


if __name__ == "__main__":
    main()
