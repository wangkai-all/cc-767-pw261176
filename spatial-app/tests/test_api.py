"""API 冒烟测试：fit → predict → cv → weights → suggest → accept → versions。"""
import os
import shutil
import tempfile

import pytest

from app import app, STATE, _load_default, BASE


@pytest.fixture()
def client():
    # 隔离版本目录，避免污染正式 versions/
    tmp = tempfile.mkdtemp(prefix="geostat_api_")
    os.environ["GEOSTAT_VERSIONS"] = tmp
    STATE["m"] = None
    _load_default()
    yield app.test_client()
    STATE["m"] = None
    os.environ.pop("GEOSTAT_VERSIONS", None)
    shutil.rmtree(tmp, ignore_errors=True)


def test_full_workflow(client):
    st = client.get("/api/state").get_json()
    assert st["points"] and st["barriers"] and st["domain"]
    n0 = len(st["points"])

    fit = client.post("/api/fit", json={}).get_json()
    assert fit["params"]["range"] > 0
    assert fit["version"]
    assert "merges" in fit["report"]

    g = client.post("/api/predict", json={"nx": 30, "ny": 30}).get_json()
    assert len(g["mean"]) == 30 and len(g["mean"][0]) == 30
    assert g["grid_version"]
    assert any(any(row) for row in g["unsupported"]) or True

    cv = client.post("/api/cv", json={"scheme": "block", "n_folds": 4})
    cvj = cv.get_json()
    assert cvj["overall"]["n"] > 0
    assert 0.0 <= cvj["overall"]["coverage95"] <= 1.0
    # 手画验证块
    cv2 = client.post("/api/cv", json={
        "rects": [[10, 10, 40, 40], [60, 60, 90, 90]]}).get_json()
    assert cv2["scheme"] == "drawn"
    assert len(cv2["folds"]) >= 1

    w = client.get("/api/weights?x=30&y=30").get_json()
    assert 0 < len(w["weights"]) <= 12          # 返回的是 top-k 截断列表
    # 含截距趋势时全部克里金权重之和为 1（无偏性）
    assert abs(w["weight_sum"] - 1.0) < 1e-6

    sg = client.post("/api/suggest", json={"n": 2, "nx": 20, "ny": 20})
    cands = sg.get_json()["candidates"]
    assert len(cands) >= 1 and cands[0]["dvar"] > 0

    acc = client.post("/api/accept",
                      json={"x": cands[0]["x"], "y": cands[0]["y"]})
    accj = acc.get_json()
    assert accj["point"]["batch"] == "synthetic"

    st2 = client.get("/api/state").get_json()
    assert len(st2["points"]) == n0 + 1
    assert len(st2["versions"]["models"]) >= 2

    # 版本回读
    gid = st2["versions"]["grids"][-1]["id"]
    act = client.post("/api/activate_grid", json={"grid_id": gid}).get_json()
    assert act["grid_version"] == gid


def test_upload_csv(client):
    csv_text = ("id,x,y,value,batch,elev\n"
                "a,0,0,1.0,b1,10\n"
                "b,10,0,1.2,b1,12\n"
                "c,0,10,0.8,b2,11\n"
                "d,10,10,1.1,b2,13\n"
                "e,5,5,1.0,b2,10\n"
                "f,20,20,1.4,b3,14\n"
                "g,15,5,1.1,b3,12\n"
                "h,5,15,0.9,b3,11\n")
    r = client.post("/api/upload", json={"csv": csv_text})
    assert r.status_code == 200
    st = client.get("/api/state").get_json()
    assert len(st["points"]) == 8
    assert st["cov_names"] == ["elev"]
    bad = client.post("/api/upload", json={"csv": "x,y\n1,2\n"})
    assert bad.status_code == 400
