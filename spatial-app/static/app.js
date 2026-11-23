/* 空间数据分析前端：栅格渲染、验证块绘制、权重追溯、建议接受、版本切换 */
"use strict";

// ---------- 调色板（固定顺序分类色；顺序 ramp 单色相） ----------
const BATCH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#6250d6", "#e34948"];
const BATCH_SHAPES = ["circle", "square", "triangle", "diamond", "cross", "star"];
const BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"];
const ORANGE_RAMP = ["#fce4d6", "#f9d3bd", "#f6bd9c", "#f2a67a", "#ee8f5c",
                     "#eb6834", "#db5a26", "#c74e1f", "#b04319", "#963814",
                     "#7c2d0f", "#61220b"];
const POS_C = "#2a78d6", NEG_C = "#e34948";

const S = {
  points: [], covNames: [], barriers: [], domain: null, config: {},
  grid: null, layer: "mean", mode: "inspect",
  rects: [], suggestions: [], weights: null,
  batchList: [], versions: null, view: null,
  modelVersion: null, gridVersion: null,
};

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const PAD = 34;

// ---------- 工具 ----------
async function api(path, method = "GET", body = null) {
  const opt = { method, headers: { "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function hexRgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16),
          parseInt(h.slice(5, 7), 16)];
}
function rampColor(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (ramp.length - 1), i = Math.floor(x), f = x - i;
  const a = hexRgb(ramp[i]), b = hexRgb(ramp[Math.min(i + 1, ramp.length - 1)]);
  return a.map((v, k) => Math.round(v + (b[k] - v) * f));
}
function batchIndex(b) {
  let i = S.batchList.indexOf(b);
  if (i < 0) { S.batchList.push(b); i = S.batchList.length - 1; }
  return i;
}
function batchColor(b) { return BATCH_COLORS[batchIndex(b) % BATCH_COLORS.length]; }
function batchShape(b) { return BATCH_SHAPES[batchIndex(b) % BATCH_SHAPES.length]; }

// ---------- 坐标变换 ----------
function computeView() {
  const xs = S.points.map(p => p.x), ys = S.points.map(p => p.y);
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (S.domain) {
    xmin = Math.min(xmin, S.domain.xmin); xmax = Math.max(xmax, S.domain.xmax);
    ymin = Math.min(ymin, S.domain.ymin); ymax = Math.max(ymax, S.domain.ymax);
  }
  const dx = (xmax - xmin) * 0.05 + 1e-6, dy = (ymax - ymin) * 0.05 + 1e-6;
  S.view = { xmin: xmin - dx, xmax: xmax + dx, ymin: ymin - dy, ymax: ymax + dy };
}
function xToPx(x) {
  return PAD + (x - S.view.xmin) / (S.view.xmax - S.view.xmin)
    * (canvas.width - 2 * PAD);
}
function yToPx(y) {
  return canvas.height - PAD - (y - S.view.ymin) / (S.view.ymax - S.view.ymin)
    * (canvas.height - 2 * PAD);
}
function pxToX(px) {
  return S.view.xmin + (px - PAD) / (canvas.width - 2 * PAD)
    * (S.view.xmax - S.view.xmin);
}
function pxToY(py) {
  return S.view.ymin + (canvas.height - PAD - py) / (canvas.height - 2 * PAD)
    * (S.view.ymax - S.view.ymin);
}

// ---------- 渲染 ----------
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fcfcfb";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (S.grid) drawRaster();
  drawBarriers();
  drawRects();
  drawPoints();
  drawSuggestions();
  drawWeights();
  drawFrame();
}

function drawRaster() {
  const g = S.grid;
  const nx = g.gx.length, ny = g.gy.length;
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(nx, ny);
  const field = S.layer === "mean" ? g.mean : g.sd;
  let mn = Infinity, mx = -Infinity;
  for (const row of field) for (const v of row) {
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const ramp = S.layer === "mean" ? BLUE_RAMP : ORANGE_RAMP;
  for (let jy = 0; jy < ny; jy++) {
    for (let jx = 0; jx < nx; jx++) {
      const t = (field[jy][jx] - mn) / (mx - mn || 1);
      const [r, gg, b] = rampColor(ramp, t);
      const k = ((ny - 1 - jy) * nx + jx) * 4;
      img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b;
      img.data[k + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const x0 = xToPx(g.gx[0]), x1 = xToPx(g.gx[nx - 1]);
  const y0 = yToPx(g.gy[ny - 1]), y1 = yToPx(g.gy[0]);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, x0, y0, x1 - x0, y1 - y0);

  // 屏障阻断格：深灰
  ctx.fillStyle = "rgba(56,56,53,0.85)";
  cellLoop(g, (jx, jy, rx, ry, rw, rh) => {
    if (g.blocked[jy][jx]) ctx.fillRect(rx, ry, rw, rh);
  });
  // 无支持区域：45° 纹理（语义标记，非装饰）
  const pat = hatchPattern();
  ctx.save();
  ctx.fillStyle = pat;
  cellLoop(g, (jx, jy, rx, ry, rw, rh) => {
    if (g.unsupported[jy][jx] && !g.blocked[jy][jx])
      ctx.fillRect(rx, ry, rw, rh);
  });
  ctx.restore();

  drawColorbar(mn, mx, ramp,
               S.layer === "mean" ? "预测均值" : "预测标准差（误差范围）");
}

function cellLoop(g, fn) {
  const nx = g.gx.length, ny = g.gy.length;
  const rw = (xToPx(g.gx[nx - 1]) - xToPx(g.gx[0])) / (nx - 1);
  const rh = (yToPx(g.gy[0]) - yToPx(g.gy[ny - 1])) / (ny - 1);
  for (let jy = 0; jy < ny; jy++)
    for (let jx = 0; jx < nx; jx++)
      fn(jx, jy, xToPx(g.gx[jx]) - rw / 2, yToPx(g.gy[jy]) - rh / 2, rw, rh);
}

function hatchPattern() {
  const p = document.createElement("canvas");
  p.width = p.height = 7;
  const c = p.getContext("2d");
  c.strokeStyle = "rgba(11,11,11,0.55)";
  c.lineWidth = 1.2;
  c.beginPath(); c.moveTo(-2, 9); c.lineTo(9, -2); c.stroke();
  c.beginPath(); c.moveTo(-2, 2); c.lineTo(2, -2); c.stroke();
  const pat = ctx.createPattern(p, "repeat");
  return pat;
}

function drawBarriers() {
  ctx.save();
  for (const poly of S.barriers) {
    ctx.beginPath();
    poly.forEach(([x, y], i) => {
      if (i === 0) ctx.moveTo(xToPx(x), yToPx(y));
      else ctx.lineTo(xToPx(x), yToPx(y));
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(11,11,11,0.18)";
    ctx.fill();
    ctx.strokeStyle = "#0b0b0b";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawShape(x, y, shape, size, color) {
  ctx.beginPath();
  if (shape === "circle") ctx.arc(x, y, size, 0, 2 * Math.PI);
  else if (shape === "square") ctx.rect(x - size, y - size, 2 * size, 2 * size);
  else if (shape === "triangle") {
    ctx.moveTo(x, y - size); ctx.lineTo(x + size, y + size);
    ctx.lineTo(x - size, y + size); ctx.closePath();
  } else if (shape === "diamond") {
    ctx.moveTo(x, y - size); ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size); ctx.lineTo(x - size, y); ctx.closePath();
  } else if (shape === "cross") {
    ctx.moveTo(x - size, y - size); ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size); ctx.lineTo(x - size, y + size);
  } else { // star
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? size * 0.45 : size;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }
  if (shape === "cross") {
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.stroke();
  } else {
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = "rgba(11,11,11,0.55)"; ctx.lineWidth = 1.2; ctx.stroke();
  }
}

function drawPoints() {
  for (const p of S.points) {
    drawShape(xToPx(p.x), yToPx(p.y), batchShape(p.batch), 5.5,
              batchColor(p.batch));
  }
}

function drawRects() {
  ctx.save();
  ctx.strokeStyle = "#52514e";
  ctx.fillStyle = "rgba(82,81,78,0.10)";
  ctx.lineWidth = 1.6;
  ctx.setLineDash([6, 4]);
  for (const [x0, y0, x1, y1] of S.rects) {
    const rx = xToPx(Math.min(x0, x1)), ry = yToPx(Math.max(y0, y1));
    const rw = Math.abs(xToPx(x1) - xToPx(x0));
    const rh = Math.abs(yToPx(y1) - yToPx(y0));
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);
  }
  ctx.restore();
}

function drawSuggestions() {
  ctx.save();
  ctx.font = "bold 11px system-ui";
  S.suggestions.forEach((s, i) => {
    const x = xToPx(s.x), y = yToPx(s.y);
    drawShape(x, y, "star", 9, "#fab219");
    ctx.fillStyle = "#0b0b0b";
    ctx.fillText(String(i + 1), x + 10, y - 8);
  });
  ctx.restore();
}

function drawWeights() {
  if (!S.weights) return;
  const w = S.weights;
  const cx = xToPx(w.x), cy = yToPx(w.y);
  const wmax = Math.max(...w.weights.map(t => Math.abs(t.w)), 1e-12);
  ctx.save();
  for (const t of w.weights) {
    const px = xToPx(t.x), py = yToPx(t.y);
    ctx.beginPath();
    ctx.moveTo(cx, cy); ctx.lineTo(px, py);
    ctx.strokeStyle = t.w >= 0 ? POS_C : NEG_C;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1 + 5 * Math.abs(t.w) / wmax;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
  ctx.fillStyle = "#0b0b0b"; ctx.fill();
  ctx.restore();
}

function drawFrame() {
  ctx.strokeStyle = "#c3c2b7";
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD, PAD, canvas.width - 2 * PAD, canvas.height - 2 * PAD);
  ctx.fillStyle = "#898781";
  ctx.font = "11px system-ui";
  ctx.fillText(S.view.xmin.toFixed(0), PAD, canvas.height - PAD + 14);
  ctx.fillText(S.view.xmax.toFixed(0), canvas.width - PAD - 20,
               canvas.height - PAD + 14);
  ctx.fillText(S.view.ymax.toFixed(0), 4, PAD + 4);
  ctx.fillText(S.view.ymin.toFixed(0), 4, canvas.height - PAD);
}

function drawColorbar(mn, mx, ramp, label) {
  const cb = document.getElementById("colorbar");
  const c = cb.getContext("2d");
  const img = c.createImageData(cb.width, cb.height);
  for (let x = 0; x < cb.width; x++) {
    const [r, g, b] = rampColor(ramp, x / (cb.width - 1));
    for (let y = 0; y < cb.height; y++) {
      const k = (y * cb.width + x) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b;
      img.data[k + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
  document.getElementById("colorbar-min").textContent = mn.toFixed(2);
  document.getElementById("colorbar-max").textContent = mx.toFixed(2);
  document.getElementById("colorbar-label").textContent = label;
}

// ---------- 面板渲染 ----------
function renderLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = S.batchList.map((b, i) =>
    `<span class="legend-item"><span class="legend-swatch" ` +
    `style="background:${BATCH_COLORS[i % BATCH_COLORS.length]}"></span>` +
    `${b || "(无批次)"}</span>`).join("");
}

function renderParams(fit) {
  const p = fit.params;
  document.getElementById("params-out").textContent =
    `模型 ${p.model} · 偏基台 ${p.sill.toFixed(3)} · 变程 ` +
    `${p.range.toFixed(1)} · 块金 ${p.nugget.toFixed(3)}\n` +
    `趋势: ` + Object.entries(fit.beta)
      .map(([k, v]) => `${k}=${v.toFixed(3)}`).join(", ") +
    `\n测点 ${fit.n_points} · 模型版本 ${fit.version || "-"}`;
}

function renderPreprocess(rep) {
  const el = document.getElementById("preprocess-out");
  const parts = [];
  if (rep.merges?.length) {
    parts.push(`<b>精确重复合并（${rep.merges.length} 组）</b><ul>` +
      rep.merges.map(m =>
        `<li>${m.ids.join("+")} @ (${m.at.map(v => v.toFixed(1))})：` +
        `${m.values.map(v => v.toFixed(2)).join(", ")} → ` +
        `<b>${m.merged_value.toFixed(2)}</b>；${m.note}</li>`).join("") +
      `</ul>`);
  }
  if (rep.near_duplicates?.length) {
    parts.push(`<b>近重合处理（${rep.near_duplicates.length} 组）</b><ul>` +
      rep.near_duplicates.map(n =>
        `<li>${n.ids.join("+")}：${n.action || ("合并为 " +
          n.merged_value.toFixed(2))}</li>`).join("") + `</ul>`);
  }
  if (rep.dropped?.length) {
    parts.push(`<b>剔除（${rep.dropped.length}）</b><ul>` +
      rep.dropped.map(d => `<li>${d.id}：${d.reason}</li>`).join("") + `</ul>`);
  }
  if (rep.conditioning && rep.conditioning.jitter) {
    const c = rep.conditioning;
    parts.push(`<b>数值病态 → 加噪稳定</b>：对角 jitter=${c.jitter.toExponential(2)}` +
      `（基台的 ${(c.relative_to_sill * 100).toExponential(1)}%），` +
      `条件数 ${fmtCond(c.condition_before)} → ${fmtCond(c.condition_after)}`);
  }
  if (rep.trend?.dropped_columns?.length) {
    parts.push(`<b>趋势列共线剔除</b>：${rep.trend.dropped_columns.join(", ")}`);
  }
  el.innerHTML = parts.length ? parts.join("")
    : "数据无需修正（无重复、无病态）。";
}
function fmtCond(c) {
  return typeof c === "number" ? c.toExponential(1) : String(c);
}

function renderCV(res) {
  const el = document.getElementById("cv-out");
  if (!res.overall) { el.innerHTML = "没有可评估的折。"; return; }
  const rows = res.folds.map(f =>
    `<tr><td>折 ${f.fold + 1}</td><td>${f.n_test}</td>` +
    `<td>${f.rmse.toFixed(3)}</td><td>${f.mae.toFixed(3)}</td>` +
    `<td>${(f.coverage90 * 100).toFixed(0)}%</td>` +
    `<td>${(f.coverage95 * 100).toFixed(0)}%</td>` +
    `<td>${f.params ? f.params.range.toFixed(1) : "-"}</td></tr>`).join("");
  const o = res.overall;
  el.innerHTML =
    `<div class="hint">方案：${res.scheme}（每折协方差参数仅由训练块重新估计）</div>` +
    `<table class="metrics"><tr><th>折</th><th>n</th><th>RMSE</th>` +
    `<th>MAE</th><th>覆盖90</th><th>覆盖95</th><th>变程</th></tr>` +
    rows +
    `<tr class="overall"><td>总体</td><td>${o.n}</td>` +
    `<td>${o.rmse.toFixed(3)}</td><td>${o.mae.toFixed(3)}</td>` +
    `<td>${(o.coverage90 * 100).toFixed(0)}%</td>` +
    `<td>${(o.coverage95 * 100).toFixed(0)}%</td><td></td></tr></table>`;
}

function renderSuggestions() {
  const el = document.getElementById("suggest-out");
  if (!S.suggestions.length) { el.innerHTML = ""; return; }
  el.innerHTML = S.suggestions.map((s, i) =>
    `<div class="suggest-row"><span class="star">★${i + 1}</span>` +
    `<span>(${s.x.toFixed(1)}, ${s.y.toFixed(1)})</span>` +
    `<span>Δvar ${s.dvar.toFixed(2)}</span>` +
    `<span>费用 ${s.cost.toFixed(2)}</span>` +
    `<button data-i="${i}" class="accept-btn">接受</button></div>`).join("");
  el.querySelectorAll(".accept-btn").forEach(btn =>
    btn.addEventListener("click", () => acceptSuggestion(+btn.dataset.i)));
}

function renderVersions(v) {
  S.versions = v;
  const el = document.getElementById("versions-out");
  const models = v.models.map(m =>
    `<div class="version-row ${m.id === S.modelVersion ? "active" : ""}">` +
    `模型 ${m.id} · ${m.model} · 变程 ${m.range.toFixed(1)} · ` +
    `${m.n_points}点 · ${m.created}</div>`).join("");
  const grids = v.grids.map(g =>
    `<div class="version-row grid-row ${g.id === S.gridVersion ? "active" : ""}" ` +
    `data-gid="${g.id}">栅格 ${g.id} ← ${g.model_id} · ` +
    `${g.nx}×${g.ny} · ${g.created}</div>`).join("");
  el.innerHTML = models + grids;
  el.querySelectorAll(".grid-row").forEach(row =>
    row.addEventListener("click", () => activateGrid(row.dataset.gid)));
}

function renderWeights() {
  const el = document.getElementById("weights-body");
  const panel = document.getElementById("weights-panel");
  if (!S.weights) { el.textContent = "尚未选择预测格。"; return; }
  panel.open = true;
  const w = S.weights;
  el.innerHTML =
    `<p>预测格 (${w.x.toFixed(1)}, ${w.y.toFixed(1)})：` +
    `均值 <b>${w.mean.toFixed(3)}</b>，标准差 <b>${w.sd.toFixed(3)}</b>，` +
    `权重和 ${w.weight_sum.toFixed(3)}</p>` +
    `<table class="metrics"><tr><th>测点</th><th>权重</th></tr>` +
    w.weights.map(t =>
      `<tr><td>${t.id}</td><td class="${t.w >= 0 ? "w-pos" : "w-neg"}">` +
      `${t.w.toFixed(4)}</td></tr>`).join("") + `</table>`;
}

function renderDataTable() {
  const el = document.getElementById("data-table-body");
  const covHeads = S.covNames.map(c => `<th>${c}</th>`).join("");
  el.innerHTML =
    `<table><tr><th>id</th><th>x</th><th>y</th><th>value</th><th>batch</th>` +
    covHeads + `</tr>` +
    S.points.map(p =>
      `<tr><td>${p.id}</td><td>${p.x.toFixed(2)}</td><td>${p.y.toFixed(2)}</td>` +
      `<td>${p.value.toFixed(3)}</td><td>${p.batch}</td>` +
      S.covNames.map(c => `<td>${p.covs[c].toFixed(2)}</td>`).join("") +
      `</tr>`).join("") + `</table>`;
}

function updateBadges() {
  document.getElementById("version-badges").innerHTML =
    `模型 <b>${S.modelVersion || "未拟合"}</b> · 栅格 <b>${S.gridVersion || "-"}</b>`;
}

// ---------- 动作 ----------
async function loadState() {
  const st = await api("/api/state");
  S.points = st.points;
  S.covNames = st.cov_names;
  S.barriers = st.barriers;
  S.domain = st.domain;
  S.config = st.config;
  S.batchList = [...new Set(st.points.map(p => p.batch))];
  S.modelVersion = st.model_version;
  S.gridVersion = st.grid_version;
  computeView();
  renderLegend();
  renderDataTable();
  renderVersions(st.versions);
  updateBadges();
  draw();
}

async function fit() {
  const cfg = {
    model: document.getElementById("cfg-model").value,
    use_xy_trend: document.getElementById("cfg-xytrend").checked,
    use_covariates: document.getElementById("cfg-covtrend").checked,
    use_barriers: document.getElementById("cfg-barriers").checked,
    near_strategy: document.getElementById("cfg-near").value,
  };
  const out = await api("/api/fit", "POST", { config: cfg });
  S.modelVersion = out.version;
  renderParams(out);
  renderPreprocess(out.report);
  updateBadges();
  return out;
}

async function predict() {
  const g = await api("/api/predict", "POST", { nx: 64, ny: 64 });
  S.grid = g;
  S.gridVersion = g.grid_version;
  S.weights = null;
  updateBadges();
  draw();
  const st = await api("/api/versions");
  renderVersions(st);
}

async function runCV(useDrawn) {
  const body = {
    scheme: document.getElementById("cv-scheme").value,
    n_folds: +document.getElementById("cv-folds").value,
  };
  if (useDrawn) {
    if (!S.rects.length) {
      document.getElementById("cv-out").innerHTML =
        '<span class="error">请先用「画验证块」在图上拖出矩形。</span>';
      return;
    }
    body.rects = S.rects;
  }
  document.getElementById("cv-out").textContent = "验证中（每折重新估计参数）…";
  const res = await api("/api/cv", "POST", body);
  renderCV(res);
}

async function suggest() {
  const body = {
    n: +document.getElementById("sg-n").value,
    base_cost: +document.getElementById("sg-base").value,
    travel_cost: +document.getElementById("sg-travel").value,
  };
  document.getElementById("suggest-out").textContent = "评分中…";
  const out = await api("/api/suggest", "POST", body);
  S.suggestions = out.candidates;
  renderSuggestions();
  draw();
}

async function acceptSuggestion(i) {
  const s = S.suggestions[i];
  const out = await api("/api/accept", "POST", { x: s.x, y: s.y });
  S.suggestions.splice(i, 1);
  await loadState();
  S.modelVersion = out.fit.version;
  renderParams(out.fit);
  renderPreprocess(out.fit.report);
  renderSuggestions();
  await predict();
}

async function activateGrid(gid) {
  const g = await api("/api/activate_grid", "POST", { grid_id: gid });
  S.grid = g;
  S.gridVersion = g.grid_version;
  updateBadges();
  draw();
  if (S.versions) renderVersions(S.versions);
}

// ---------- 画布交互 ----------
let dragStart = null;
canvas.addEventListener("mousedown", e => {
  if (S.mode !== "draw") return;
  const r = canvas.getBoundingClientRect();
  dragStart = [pxToX(e.clientX - r.left), pxToY(e.clientY - r.top)];
});
canvas.addEventListener("mouseup", e => {
  if (S.mode !== "draw" || !dragStart) return;
  const r = canvas.getBoundingClientRect();
  const p = [pxToX(e.clientX - r.left), pxToY(e.clientY - r.top)];
  if (Math.abs(p[0] - dragStart[0]) > 1 && Math.abs(p[1] - dragStart[1]) > 1) {
    S.rects.push([dragStart[0], dragStart[1], p[0], p[1]]);
  }
  dragStart = null;
  draw();
});
canvas.addEventListener("click", async e => {
  if (S.mode !== "inspect" || !S.grid) return;
  const r = canvas.getBoundingClientRect();
  const x = pxToX(e.clientX - r.left), y = pxToY(e.clientY - r.top);
  const w = await api(`/api/weights?x=${x}&y=${y}`);
  S.weights = { x, y, ...w };
  renderWeights();
  draw();
});
canvas.addEventListener("mousemove", e => {
  const tip = document.getElementById("tooltip");
  const r = canvas.getBoundingClientRect();
  const x = pxToX(e.clientX - r.left), y = pxToY(e.clientY - r.top);
  let text = `(${x.toFixed(1)}, ${y.toFixed(1)})`;
  if (S.grid) {
    const g = S.grid;
    const jx = nearestIdx(g.gx, x), jy = nearestIdx(g.gy, y);
    if (jx != null) {
      if (g.blocked[jy][jx]) text += " 屏障内";
      else if (g.unsupported[jy][jx]) text += " 无支持区域";
      else text += ` 均值 ${g.mean[jy][jx].toFixed(2)} · ` +
                   `sd ${g.sd[jy][jx].toFixed(2)}`;
    }
  }
  tip.textContent = text;
  tip.hidden = false;
  tip.style.left = (e.clientX - r.left + 12) + "px";
  tip.style.top = (e.clientY - r.top + 12) + "px";
});
canvas.addEventListener("mouseleave", () => {
  document.getElementById("tooltip").hidden = true;
});
function nearestIdx(arr, v) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - v);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// ---------- 控件绑定 ----------
document.querySelectorAll("input[name=layer]").forEach(r =>
  r.addEventListener("change", e => { S.layer = e.target.value; draw(); }));
document.querySelectorAll("input[name=mode]").forEach(r =>
  r.addEventListener("change", e => { S.mode = e.target.value; }));
document.getElementById("btn-fit").addEventListener("click", async () => {
  try { await fit(); await predict(); } catch (e) { showErr("params-out", e); }
});
document.getElementById("btn-predict").addEventListener("click", async () => {
  try { await predict(); } catch (e) { showErr("params-out", e); }
});
document.getElementById("btn-cv").addEventListener("click", () =>
  runCV(false).catch(e => showErr("cv-out", e)));
document.getElementById("btn-cv-drawn").addEventListener("click", () =>
  runCV(true).catch(e => showErr("cv-out", e)));
document.getElementById("btn-clear-rects").addEventListener("click", () => {
  S.rects = []; draw();
});
document.getElementById("btn-suggest").addEventListener("click", () =>
  suggest().catch(e => showErr("suggest-out", e)));
document.getElementById("file-upload").addEventListener("change", async e => {
  const f = e.target.files[0];
  if (!f) return;
  const text = await f.text();
  try {
    await api("/api/upload", "POST", { csv: text });
    await loadState();
    await fit();
    await predict();
  } catch (err) { alert("导入失败: " + err.message); }
});
function showErr(id, e) {
  document.getElementById(id).innerHTML =
    `<span class="error">${e.message}</span>`;
}

// ---------- 启动 ----------
(async () => {
  await loadState();
  try {
    await fit();
    await predict();
  } catch (e) {
    console.error(e);
  }
})();
