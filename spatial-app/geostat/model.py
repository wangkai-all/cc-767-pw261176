"""顶层模型：预处理 → 参数估计（可限定训练块）→ 泛克里金 → 栅格/验证/设计。

并负责模型与网格版本落盘（versions/ 目录）。
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from . import blocks as blk
from . import covariance as cov
from . import design as dsg
from .dataset import Dataset
from .estimation import estimate_params
from .geom import BarrierSet
from .kriging import (KrigingParams, UniversalKriging, apply_trend_transform,
                      drop_collinear, trend_matrix)
from .preprocessing import (PreprocessReport, handle_near_duplicates,
                            merge_exact_duplicates)
from .simulate import conditional_draw

DEFAULT_CONFIG = {
    "model": "matern32",          # 相关模型
    "use_xy_trend": True,         # 坐标线性趋势
    "use_covariates": True,       # 地形协变量趋势
    "use_barriers": True,         # 屏障阻断相关
    "near_strategy": "merge",     # 近重合点: merge | jitter
    "near_eps": 0.5,              # 近重合阈值（坐标单位）
    "unsupported_corr": 0.05,     # 最大相关低于此值 → 无支持
}


# ------------------------------------------------------------ 版本存储

class VersionStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {"models": [], "grids": []}
            self._save()

    def _save(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=1)

    def _next_id(self, kind):
        seq = [e["id"] for e in self.index[kind]]
        n = 1 + max([int(s.rsplit("_", 1)[1]) for s in seq], default=0)
        return f"{kind[:-1]}_{n:04d}"

    def save_model(self, payload):
        mid = self._next_id("models")
        payload = dict(payload, id=mid, created=time.strftime("%F %T"))
        with open(os.path.join(self.root, f"{mid}.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        self.index["models"].append({
            "id": mid, "created": payload["created"],
            "model": payload["params"]["model"],
            "range": payload["params"]["range"],
            "sill": payload["params"]["sill"],
            "nugget": payload["params"]["nugget"],
            "n_points": payload["n_points"],
            "data_hash": payload["data_hash"],
        })
        self._save()
        return mid

    def save_grid(self, model_id, arrays, meta):
        gid = self._next_id("grids")
        np.savez_compressed(os.path.join(self.root, f"{gid}.npz"), **arrays)
        entry = dict(meta, id=gid, model_id=model_id,
                     created=time.strftime("%F %T"))
        self.index["grids"].append(entry)
        self._save()
        return gid

    def load_grid(self, gid):
        path = os.path.join(self.root, f"{gid}.npz")
        z = np.load(path)
        return {k: z[k] for k in z.files}

    def load_model(self, mid):
        with open(os.path.join(self.root, f"{mid}.json")) as f:
            return json.load(f)


# ------------------------------------------------------------ 主模型

class SpatialModel:
    def __init__(self, dataset: Dataset, barriers=None,
                 versions_dir="versions", config=None, domain=None):
        self.dataset = dataset
        dom = dataset.domain
        # 分析区域：默认数据包围盒，可由 domain=(xmin,ymin,xmax,ymax) 指定
        if domain is not None:
            self.domain = {"xmin": domain[0], "ymin": domain[1],
                           "xmax": domain[2], "ymax": domain[3]}
        else:
            self.domain = dom
        pad = 0.05 * max(dom["xmax"] - dom["xmin"],
                         dom["ymax"] - dom["ymin"], 1e-9)
        bounds = (self.domain["xmin"] - pad, self.domain["ymin"] - pad,
                  self.domain["xmax"] + pad, self.domain["ymax"] + pad)
        self.barriers = BarrierSet(barriers, bounds=bounds)
        self.config = dict(DEFAULT_CONFIG, **(config or {}))
        self.store = VersionStore(versions_dir)
        # 预处理后的数据
        self.coords = None
        self.values = None
        self.covs = None
        self.batches = None
        self.ids = None
        self.counts = None
        self.report = PreprocessReport()
        self._D = None            # 测点间距离（含屏障）
        # 拟合结果
        self.params = None
        self.transform = None
        self.trend_names = None
        self.krig = None
        self.model_version = None
        self.grid_version = None
        self.grid_cache = None

    # -------------------------------------------------- 数据与距离

    def data_hash(self):
        h = hashlib.sha1()
        for arr in (self.dataset.coords, self.dataset.values):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update("|".join(map(str, self.dataset.batches)).encode())
        h.update(json.dumps(self.config, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def preprocess(self):
        ds = self.dataset
        self.report = PreprocessReport()
        covs = ds.covariates if ds.covariates.size else np.zeros((len(ds), 0))
        coords, values, covs, batches, ids, counts = merge_exact_duplicates(
            ds.coords, ds.values, covs, ds.batches, ds.ids, self.report)
        (coords, values, covs, batches, ids, counts) = handle_near_duplicates(
            coords, values, covs, batches, ids, counts,
            self.config["near_eps"], self.config["near_strategy"],
            self.report)
        # 落入屏障内的点：丢弃并报告
        if self.barriers:
            inside = np.zeros(len(values), dtype=bool)
            for poly in self.barriers.polygons:
                from .geom import point_in_polygon
                inside |= np.array([point_in_polygon(c, poly)
                                    for c in coords])
            if inside.any():
                for i in np.where(inside)[0]:
                    self.report.dropped.append({
                        "id": str(ids[i]),
                        "at": [float(coords[i, 0]), float(coords[i, 1])],
                        "reason": "测点落入屏障多边形内，已剔除"})
                keep = ~inside
                coords, values, covs, batches, ids, counts = (
                    coords[keep], values[keep], covs[keep],
                    batches[keep], ids[keep], counts[keep])
        self.coords, self.values, self.covs = coords, values, covs
        self.batches, self.ids, self.counts = batches, ids, counts
        self._D = None
        return self.report

    def distance_matrix(self):
        if self._D is None:
            if self.config["use_barriers"] and self.barriers:
                self._D = self.barriers.pairwise(self.coords)
            else:
                self._D = BarrierSet(None).pairwise(self.coords)
        return self._D

    def _trend(self, coords, covs, fit=False):
        use_cov = self.config["use_covariates"] and covs is not None \
            and np.asarray(covs).size > 0
        c = covs if use_cov else None
        if fit:
            F, tr = trend_matrix(coords, c, self.config["use_xy_trend"])
            self.transform = tr
            return F
        return apply_trend_transform(coords, c, self.transform)

    # -------------------------------------------------- 拟合

    def fit(self, config=None, train_idx=None, save_version=True):
        """拟合：预处理 → 趋势 → 参数估计（仅 train_idx 给定时的训练块）。"""
        if config:
            self.config.update(config)
            self._D = None
        self.preprocess()
        D = self.distance_matrix()
        F = self._trend(self.coords, self.covs, fit=True)
        F, kept = drop_collinear(F, self.transform["names"], self.report)
        self.trend_names = kept
        self._F_full = F

        idx = np.arange(len(self.values)) if train_idx is None \
            else np.asarray(train_idx)
        Dtr = D[np.ix_(idx, idx)]
        params, diag = estimate_params(
            Dtr, F[idx], self.values[idx], self.config["model"],
            counts=self.counts[idx])
        self.params = params
        self.krig = UniversalKriging(
            Dtr, F[idx], self.values[idx],
            KrigingParams(params["model"], params["sill"],
                          params["range"], params["nugget"]),
            counts=self.counts[idx], report=self.report)
        self._train_idx = idx
        if save_version and train_idx is None:
            self.model_version = self.store.save_model({
                "config": self.config,
                "params": params,
                "trend": {"names": kept, "transform": self.transform,
                          "beta": list(map(float, self.krig.beta))},
                "preprocess": self.report.to_dict(),
                "n_points": int(len(idx)),
                "data_hash": self.data_hash(),
                "estimation": diag,
            })
        return {"params": params, "report": self.report.to_dict(),
                "version": self.model_version,
                "beta": dict(zip(kept, map(float, self.krig.beta))),
                "n_points": int(len(idx))}

    # -------------------------------------------------- 栅格预测

    def predict_grid(self, nx=60, ny=60, save_version=True):
        assert self.krig is not None, "先调用 fit()"
        dom = self.domain
        pad_x = 0.03 * (dom["xmax"] - dom["xmin"] + 1e-9)
        pad_y = 0.03 * (dom["ymax"] - dom["ymin"] + 1e-9)
        gx = np.linspace(dom["xmin"] - pad_x, dom["xmax"] + pad_x, nx)
        gy = np.linspace(dom["ymin"] - pad_y, dom["ymax"] + pad_y, ny)

        idx = self._train_idx
        train_coords = self.coords[idx]
        if self.config["use_barriers"] and self.barriers:
            Dg, blocked, cell_xy = self.barriers.points_to_grid(
                train_coords, gx, gy)
        else:
            Dg, blocked, cell_xy = BarrierSet(None).points_to_grid(
                train_coords, gx, gy)
        # 网格协变量未知：以训练点协变量均值作为趋势外推基准
        cov_grid = np.tile(self.covs.mean(0), (len(cell_xy), 1)) \
            if self.covs.size else None
        Fg = self._trend(cell_xy, cov_grid)
        Fg = Fg[:, [self.transform["names"].index(n) for n in
                    self.trend_names]]
        out = self.krig.predict(Dg, Fg, target="field")
        mean, sd = out["mean"], out["sd"]

        # 无支持区域：与所有训练点的最大相关低于阈值（含被屏障隔离）
        corr_max = cov.corr(Dg, self.params["model"],
                            self.params["range"]).max(axis=0)
        corr_max = np.where(np.isfinite(Dg).any(axis=0), corr_max, 0.0)
        unsupported = corr_max < self.config["unsupported_corr"]

        arrays = {"gx": gx, "gy": gy, "mean": mean.reshape(ny, nx),
                  "sd": sd.reshape(ny, nx),
                  "unsupported": unsupported.reshape(ny, nx),
                  "blocked": blocked.reshape(ny, nx)}
        self.grid_cache = arrays
        if save_version:
            self.grid_version = self.store.save_grid(
                self.model_version, arrays,
                {"nx": nx, "ny": ny, "n_unsupported": int(unsupported.sum()),
                 "n_blocked": int(blocked.sum())})
        return arrays

    # -------------------------------------------------- 单点权重

    def weights_at(self, x, y, top=12):
        assert self.krig is not None, "先调用 fit()"
        idx = self._train_idx
        train_coords = self.coords[idx]
        pt = np.array([[x, y]], dtype=float)
        if self.config["use_barriers"] and self.barriers:
            # 单点：与训练点一起做可见性图测地距离
            D_all = self.barriers.pairwise(np.vstack([train_coords, pt]))
            D_row = D_all[:-1, -1:]                 # (n_train, 1)
        else:
            D_row = np.sqrt(((train_coords - pt)**2).sum(1))[:, None]
        cov_pt = self.covs.mean(0, keepdims=True) if self.covs.size else None
        F0 = self._trend(pt, cov_pt)
        F0 = F0[:, [self.transform["names"].index(n)
                    for n in self.trend_names]]
        out = self.krig.predict(D_row, F0, target="field",
                                return_weights=True)
        w = out["weights"][:, 0]
        order = np.argsort(-np.abs(w))[:top]
        return {
            "mean": float(out["mean"][0]), "sd": float(out["sd"][0]),
            "weights": [{
                "id": str(self.ids[idx][i]),
                "x": float(train_coords[i, 0]),
                "y": float(train_coords[i, 1]),
                "w": float(w[i]),
            } for i in order],
            "weight_sum": float(w.sum()),
        }

    # -------------------------------------------------- 交叉验证

    def cross_validate(self, scheme="block", n_folds=5, rects=None,
                       seed=0):
        """scheme: 'block' 连续空间块 | 'random' 随机拆分（对照）。"""
        self.preprocess()
        D = self.distance_matrix()
        F = self._trend(self.coords, self.covs, fit=True)
        F, kept = drop_collinear(F, self.transform["names"], self.report)
        n = len(self.values)

        if rects:
            block_ids = blk.blocks_from_rects(self.coords, rects)
            folds = blk.make_folds(block_ids)
        elif scheme == "random":
            block_ids = blk.random_split_ids(n, n_folds, seed)
            folds = blk.make_folds(block_ids, n_folds)
        else:
            block_ids = blk.assign_blocks(self.coords, self.batches,
                                          n_folds, seed)
            folds = blk.make_folds(block_ids, n_folds)

        def fit_fn(train_idx, test_idx):
            params, _ = estimate_params(
                D[np.ix_(train_idx, train_idx)], F[train_idx],
                self.values[train_idx], self.config["model"],
                counts=self.counts[train_idx])
            k = UniversalKriging(
                D[np.ix_(train_idx, train_idx)], F[train_idx],
                self.values[train_idx],
                KrigingParams(params["model"], params["sill"],
                              params["range"], params["nugget"]),
                counts=self.counts[train_idx])
            out = k.predict(D[np.ix_(train_idx, test_idx)], F[test_idx],
                            target="measurement")
            out["params"] = params
            return out

        result = blk.cross_validate(self.coords, self.covs, self.values,
                                    self.batches, self.counts, fit_fn,
                                    block_ids)
        result["scheme"] = scheme if not rects else "drawn"
        result["block_ids"] = [int(b) for b in block_ids]
        return result

    # -------------------------------------------------- 加点设计

    def suggest(self, n_suggest=5, base_cost=1.0, travel_cost=0.02,
                nx=40, ny=40, min_dist_factor=0.2, seed=0):
        """返回按 单位费用方差下降 排序的候选点（贪心序贯）。"""
        assert self.krig is not None, "先调用 fit()"
        idx = np.array(self._train_idx)
        dom = self.domain
        gx = np.linspace(dom["xmin"], dom["xmax"], nx)
        gy = np.linspace(dom["ymin"], dom["ymax"], ny)

        # 当前系统状态的副本（贪心时逐步加入伪测点）
        cur_coords = self.coords[idx].copy()
        cur_counts = self.counts[idx].astype(float).copy()
        cur_F = self._F_full[idx].copy()

        def build_krig(coords, F, counts):
            if self.config["use_barriers"] and self.barriers:
                D = self.barriers.pairwise(coords)
            else:
                D = BarrierSet(None).pairwise(coords)
            return UniversalKriging(
                D, F, np.zeros(len(coords)),
                KrigingParams(self.params["model"], self.params["sill"],
                              self.params["range"], self.params["nugget"]),
                counts=counts), D

        picks = []
        for _ in range(n_suggest):
            krig0, Dcur = build_krig(cur_coords, cur_F, cur_counts)
            if self.config["use_barriers"] and self.barriers:
                Dg, blocked, cell_xy = self.barriers.points_to_grid(
                    cur_coords, gx, gy)
                graph, _, _ = self.barriers.grid_graph(gx, gy)
            else:
                Dg, blocked, cell_xy = BarrierSet(None).points_to_grid(
                    cur_coords, gx, gy)
                graph = None
            cov_grid = np.tile(self.covs.mean(0), (len(cell_xy), 1)) \
                if self.covs.size else None
            Fg = self._trend(cell_xy, cov_grid)
            Fg = Fg[:, [self.transform["names"].index(n)
                        for n in self.trend_names]]
            cells_cache = krig0.trend_residual(Dg, Fg)

            # 候选 = 未阻断格点，且离现有测点不太近
            dmin = Dg.min(axis=0)
            cand_mask = (~blocked) & np.isfinite(dmin) & (
                dmin > min_dist_factor * self.params["range"])
            cand_idx = np.where(cand_mask)[0]
            if len(cand_idx) == 0:
                break
            cand_xy = cell_xy[cand_idx]
            D_cand_train = Dg[:, cand_idx].T          # (m, n_train)
            F_cand = Fg[cand_idx]
            # 候选格 → 全部目标格 的距离
            if graph is not None:
                from scipy.sparse.csgraph import dijkstra
                D_cand_cells = dijkstra(graph, directed=False,
                                        indices=cand_idx)
            else:
                d = cand_xy[:, None, :] - cell_xy[None, :, :]
                D_cand_cells = np.sqrt((d**2).sum(-1))
            dvar, var_ss = dsg.integrated_variance_reduction(
                krig0, self.params, D_cand_train, F_cand,
                D_cand_cells, cells_cache, self.params["nugget"])
            cost = dsg.candidate_cost(D_cand_train, base_cost, travel_cost)
            score = dvar / cost
            best = int(np.argmax(score))
            pick = {
                "x": float(cand_xy[best, 0]), "y": float(cand_xy[best, 1]),
                "dvar": float(dvar[best]), "cost": float(cost[best]),
                "score": float(score[best]),
                "var_at_pick": float(var_ss[best]),
            }
            picks.append(pick)
            # 伪测点并入系统
            cur_coords = np.vstack([cur_coords, cand_xy[best]])
            cur_F = np.vstack([cur_F, F_cand[best]])
            cur_counts = np.append(cur_counts, 1.0)
        return {"candidates": picks,
                "cost_model": {"base": base_cost, "travel": travel_cost}}

    # -------------------------------------------------- 接受建议 → 合成测量

    def accept_point(self, x, y, seed=None):
        """在 (x,y) 补入一次合成测量（条件模拟），加入数据集并重拟合。"""
        assert self.krig is not None, "先调用 fit()"
        w = self.weights_at(x, y)
        rng = np.random.default_rng(seed)
        value = conditional_draw(w["mean"], w["sd"]**2,
                                 self.params["nugget"], rng)
        ds = self.dataset
        new_id = f"syn_{len(ds)}"
        if ds.covariates.size:
            new_covs = np.vstack([ds.covariates,
                                  ds.covariates.mean(0, keepdims=True)])
        else:
            new_covs = np.zeros((len(ds) + 1, 0))
        self.dataset = Dataset(
            np.append(ds.ids, new_id),
            np.vstack([ds.coords, [[x, y]]]),
            np.append(ds.values, value),
            np.append(ds.batches, "synthetic"),
            new_covs,
            list(ds.cov_names),
        )
        fit_out = self.fit()
        return {"point": {"id": new_id, "x": float(x), "y": float(y),
                          "value": value, "batch": "synthetic"},
                "fit": fit_out}
