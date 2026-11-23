"""Flask API + 静态页面。运行: python run.py  （或 flask --app app run）"""
import json
import os
import traceback

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from geostat.covariance import MODELS
from geostat.dataset import load_csv
from geostat.model import SpatialModel

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder="static", static_url_path="/static")

STATE = {"m": None}


def _load_default():
    with open(os.path.join(BASE, "data", "points.csv")) as f:
        ds = load_csv(f.read())
    bpath = os.path.join(BASE, "data", "barriers.json")
    barriers, domain = [], None
    if os.path.exists(bpath):
        with open(bpath) as f:
            bj = json.load(f)
        barriers = bj.get("polygons", [])
        if bj.get("domain"):
            domain = tuple(bj["domain"])
    vdir = os.environ.get("GEOSTAT_VERSIONS",
                          os.path.join(BASE, "versions"))
    STATE["m"] = SpatialModel(ds, barriers=barriers, domain=domain,
                              versions_dir=vdir)


def _m():
    if STATE["m"] is None:
        _load_default()
    return STATE["m"]


def _err(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
    wrapper.__name__ = fn.__name__
    return wrapper


@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.get("/api/state")
@_err
def state():
    m = _m()
    return jsonify({
        "points": m.dataset.to_points(),
        "cov_names": m.dataset.cov_names,
        "barriers": [p.tolist() for p in m.barriers.polygons],
        "domain": m.domain,
        "config": m.config,
        "models": MODELS,
        "versions": m.store.index,
        "model_version": m.model_version,
        "grid_version": m.grid_version,
        "params": m.params,
    })


@app.post("/api/config")
@_err
def set_config():
    m = _m()
    cfg = request.get_json(force=True)
    allowed = {"model", "use_xy_trend", "use_covariates", "use_barriers",
               "near_strategy", "near_eps", "unsupported_corr"}
    m.config.update({k: v for k, v in cfg.items() if k in allowed})
    m._D = None
    return jsonify({"config": m.config})


@app.post("/api/fit")
@_err
def fit():
    m = _m()
    body = request.get_json(force=True) or {}
    out = m.fit(config=body.get("config"))
    return jsonify(out)


@app.post("/api/predict")
@_err
def predict():
    m = _m()
    body = request.get_json(force=True) or {}
    g = m.predict_grid(nx=int(body.get("nx", 60)), ny=int(body.get("ny", 60)))
    return jsonify({
        "gx": g["gx"].tolist(), "gy": g["gy"].tolist(),
        "mean": g["mean"].tolist(), "sd": g["sd"].tolist(),
        "unsupported": g["unsupported"].astype(int).tolist(),
        "blocked": g["blocked"].astype(int).tolist(),
        "grid_version": m.grid_version,
        "model_version": m.model_version,
    })


@app.post("/api/cv")
@_err
def cv():
    m = _m()
    body = request.get_json(force=True) or {}
    out = m.cross_validate(
        scheme=body.get("scheme", "block"),
        n_folds=int(body.get("n_folds", 5)),
        rects=body.get("rects") or None,
        seed=int(body.get("seed", 0)))
    return jsonify(out)


@app.get("/api/weights")
@_err
def weights():
    m = _m()
    x = float(request.args["x"])
    y = float(request.args["y"])
    return jsonify(m.weights_at(x, y))


@app.post("/api/suggest")
@_err
def suggest():
    m = _m()
    body = request.get_json(force=True) or {}
    out = m.suggest(
        n_suggest=int(body.get("n", 5)),
        base_cost=float(body.get("base_cost", 1.0)),
        travel_cost=float(body.get("travel_cost", 0.02)),
        nx=int(body.get("nx", 40)), ny=int(body.get("ny", 40)))
    return jsonify(out)


@app.post("/api/accept")
@_err
def accept():
    m = _m()
    body = request.get_json(force=True)
    out = m.accept_point(float(body["x"]), float(body["y"]),
                         seed=body.get("seed"))
    return jsonify(out)


@app.post("/api/upload")
@_err
def upload():
    body = request.get_json(force=True)
    ds = load_csv(body["csv"])
    m = _m()
    barriers = [p.tolist() for p in m.barriers.polygons]
    vdir = os.environ.get("GEOSTAT_VERSIONS",
                          os.path.join(BASE, "versions"))
    STATE["m"] = SpatialModel(ds, barriers=barriers, versions_dir=vdir)
    return jsonify({"n": len(ds), "cov_names": ds.cov_names})


@app.get("/api/versions")
@_err
def versions():
    return jsonify(_m().store.index)


@app.post("/api/activate_grid")
@_err
def activate_grid():
    m = _m()
    body = request.get_json(force=True)
    g = m.store.load_grid(body["grid_id"])
    m.grid_version = body["grid_id"]
    m.grid_cache = g
    return jsonify({
        "gx": g["gx"].tolist(), "gy": g["gy"].tolist(),
        "mean": g["mean"].tolist(), "sd": g["sd"].tolist(),
        "unsupported": g["unsupported"].astype(int).tolist(),
        "blocked": g["blocked"].astype(int).tolist(),
        "grid_version": body["grid_id"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
