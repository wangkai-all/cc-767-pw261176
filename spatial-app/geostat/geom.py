"""屏障（遮挡边界）与屏障感知的距离。

屏障是多边形列表。两点间若直线穿过屏障，则相关不再按欧氏距离衰减，
而按"绕过屏障的最短路径"（测地距离）衰减；完全被屏障隔离的点对距离为
inf，相关为 0 —— 因此不会跨过配置的边界平滑。

- 点-点距离：精确可见性图 + Dijkstra（样本数小，精确计算）。
- 点-网格距离：网格 8 邻接图上的多源 Dijkstra（屏障格阻断）。
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------- 基本几何

def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_proper_intersect(p1, p2, p3, p4) -> bool:
    """严格相交（不含端点触碰）。"""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and \
        d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def point_in_polygon(p, poly) -> bool:
    """射线法。poly: (k,2) 数组。"""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def segment_crosses_polygon(p, q, poly) -> bool:
    """线段是否穿过多边形内部（贴边/过顶点不算）。"""
    n = len(poly)
    for i in range(n):
        if _seg_proper_intersect(p, q, poly[i], poly[(i + 1) % n]):
            return True
    # 两端都在内部也算穿过
    return point_in_polygon(p, poly) and point_in_polygon(q, poly)


# ---------------------------------------------------------------- 屏障集合

class BarrierSet:
    """一组屏障多边形；提供测地距离。

    bounds = (xmin, ymin, xmax, ymax) 可选：插值区域边界。边界之外的
    屏障顶点不可通行——因此一条两端都接到区域边界的墙会把区域切开，
    两侧点对距离为 inf（完全不相关）。
    """

    def __init__(self, polygons: list | None, bounds=None):
        self.polygons = [np.asarray(p, dtype=float) for p in (polygons or [])]
        self.polygons = [p for p in self.polygons if len(p) >= 3]
        self.bounds = bounds

    def __bool__(self):
        return len(self.polygons) > 0

    def _usable_vertices(self):
        verts = [v for poly in self.polygons for v in poly]
        if self.bounds is not None:
            x0, y0, x1, y1 = self.bounds
            eps = 1e-9 * max(1.0, x1 - x0, y1 - y0)
            verts = [v for v in verts
                     if x0 - eps <= v[0] <= x1 + eps
                     and y0 - eps <= v[1] <= y1 + eps]
        return verts

    # ------------------------------------------------------------ 点-点

    def pairwise(self, pts: np.ndarray) -> np.ndarray:
        """(n,2) -> (n,n) 测地距离矩阵。无屏障时退化为欧氏距离。"""
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        if not self.polygons:
            d = pts[:, None, :] - pts[None, :, :]
            return np.sqrt((d**2).sum(-1))

        verts = self._usable_vertices()
        nodes = np.vstack([pts, np.asarray(verts)]) if verts else pts.copy()
        m = len(nodes)
        adj: list[list[tuple[int, float]]] = [[] for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                if not any(segment_crosses_polygon(nodes[i], nodes[j], p)
                           for p in self.polygons):
                    w = float(np.hypot(*(nodes[i] - nodes[j])))
                    adj[i].append((j, w))
                    adj[j].append((i, w))

        D = np.full((n, n), np.inf)
        for s in range(n):
            dist = [np.inf] * m
            dist[s] = 0.0
            pq = [(0.0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            D[s] = dist[:n]
        return D

    # ------------------------------------------------------------ 点-网格

    def grid_graph(self, gx: np.ndarray, gy: np.ndarray):
        """构建网格 8 邻接图，返回 (graph_csr, blocked_mask, cell_xy)。

        blocked_mask: 位于屏障内的格点（True = 阻断）。
        """
        GX, GY = np.meshgrid(gx, gy)  # 行主序: (ny, nx)
        cell_xy = np.column_stack([GX.ravel(), GY.ravel()])
        ny, nx = len(gy), len(gx)
        blocked = np.zeros(ny * nx, dtype=bool)
        for poly in self.polygons:
            inside = np.array([point_in_polygon(c, poly) for c in cell_xy])
            blocked |= inside

        dx = float(gx[1] - gx[0]) if nx > 1 else 1.0
        dy = float(gy[1] - gy[0]) if ny > 1 else 1.0
        rows, cols, vals = [], [], []

        def idx(i, j):
            return i * nx + j

        for i in range(ny):
            for j in range(nx):
                u = idx(i, j)
                if blocked[u]:
                    continue
                for di, dj, w in ((0, 1, dx), (1, 0, dy),
                                  (1, 1, np.hypot(dx, dy)),
                                  (1, -1, np.hypot(dx, dy))):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx:
                        v = idx(ni, nj)
                        if not blocked[v]:
                            rows += [u, v]
                            cols += [v, u]
                            vals += [w, w]
        n_cells = ny * nx
        graph = sparse.csr_matrix((vals, (rows, cols)),
                                  shape=(n_cells, n_cells))
        return graph, blocked, cell_xy

    def points_to_grid(self, pts: np.ndarray, gx: np.ndarray,
                       gy: np.ndarray):
        """(n,2) 点到全部网格点的测地距离 (n, n_cells)。

        无屏障时直接欧氏距离。有屏障时把每个点接入其附近的自由格，
        在网格图上跑多源 Dijkstra。
        """
        pts = np.asarray(pts, dtype=float)
        graph, blocked, cell_xy = self.grid_graph(gx, gy)
        n_cells = len(cell_xy)
        if not self.polygons:
            d = pts[:, None, :] - cell_xy[None, :, :]
            return np.sqrt((d**2).sum(-1)), blocked, cell_xy

        n = len(pts)
        # 扩展图：格点 + 每个样本点一个节点
        rows, cols, vals = [], [], []
        dx = float(gx[1] - gx[0]) if len(gx) > 1 else 1.0
        dy = float(gy[1] - gy[0]) if len(gy) > 1 else 1.0
        r0 = 1.6 * max(dx, dy)
        free = np.where(~blocked)[0]
        for k, p in enumerate(pts):
            node = n_cells + k
            d2 = ((cell_xy[free] - p) ** 2).sum(1)
            near = free[d2 <= r0**2]
            for c in near:
                if not any(segment_crosses_polygon(p, cell_xy[c], poly)
                           for poly in self.polygons):
                    w = float(np.hypot(*(p - cell_xy[c])))
                    rows += [node, c]
                    cols += [c, node]
                    vals += [w, w]
        N = n_cells + n
        ext = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(N, N)) + sparse.block_diag(
            (graph, sparse.csr_matrix((n, n))), format="csr")
        D = dijkstra(ext, directed=False,
                     indices=np.arange(n_cells, N))[:, :n_cells]
        return D, blocked, cell_xy
