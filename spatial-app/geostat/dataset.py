"""数据集加载与校验。

CSV 格式（逗号分隔，含表头）：
    id,x,y,value,batch,<协变量列...>
- id: 字符串（可重复——重复坐标合并后 id 以 "+" 连接）
- x,y: 平面坐标（任意投影，单位自定）
- value: 观测值
- batch: 采样批次号（同批测点在验证划分中不拆分）
- 其余列一律视为地形协变量（数值型），参与趋势项
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np

REQUIRED = ("x", "y", "value")


@dataclass
class Dataset:
    ids: np.ndarray          # (n,) str
    coords: np.ndarray       # (n,2)
    values: np.ndarray       # (n,)
    batches: np.ndarray      # (n,) str
    covariates: np.ndarray   # (n,k) 可能为空 (n,0)
    cov_names: list = field(default_factory=list)

    def __len__(self):
        return len(self.values)

    @property
    def domain(self):
        return {
            "xmin": float(self.coords[:, 0].min()),
            "xmax": float(self.coords[:, 0].max()),
            "ymin": float(self.coords[:, 1].min()),
            "ymax": float(self.coords[:, 1].max()),
        }

    def to_points(self):
        pts = []
        for i in range(len(self)):
            pts.append({
                "id": str(self.ids[i]),
                "x": float(self.coords[i, 0]),
                "y": float(self.coords[i, 1]),
                "value": float(self.values[i]),
                "batch": str(self.batches[i]),
                "covs": {n: float(self.covariates[i, j])
                         for j, n in enumerate(self.cov_names)},
            })
        return pts


def load_csv(text: str) -> Dataset:
    reader = csv.DictReader(io.StringIO(text.strip()))
    cols = reader.fieldnames or []
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}；需要 x,y,value"
                         "（另可选 id,batch 与任意协变量列）")
    ids, xs, ys, vs, bs = [], [], [], [], []
    cov_names = [c for c in cols
                 if c not in ("id", "x", "y", "value", "batch")]
    covs = []
    for row_i, row in enumerate(reader):
        try:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            vs.append(float(row["value"]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"第 {row_i + 2} 行数值解析失败: {e}") from e
        ids.append(row.get("id") or f"p{row_i}")
        bs.append(row.get("batch") or "")
        covs.append([float(row[c]) if row.get(c) not in (None, "") else 0.0
                     for c in cov_names])
    if not xs:
        raise ValueError("CSV 没有数据行")
    coords = np.column_stack([xs, ys])
    if not np.isfinite(coords).all() or not np.isfinite(vs).all():
        raise ValueError("坐标或观测值含非有限数")
    cov_arr = np.array(covs, dtype=float).reshape(len(xs), len(cov_names))
    return Dataset(np.array(ids, dtype=object), coords,
                   np.array(vs, dtype=float), np.array(bs, dtype=object),
                   cov_arr, list(cov_names))
