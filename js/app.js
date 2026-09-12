import { standardize, pca, kmeans, pearson, quantile } from "./ml.js";

// Palette roughly follows the article's map: green, purple, blue, red, yellow, then extras.
const PALETTE = ["#4fc47f", "#a06cd5", "#4f8fe0", "#e8544f", "#f2c94c", "#f28c38", "#3ac7c7", "#f078b4", "#a67c52", "#9aa3b2", "#c6e04a", "#7b8cf0"];
const VIRIDIS = ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];

const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const fmt = (v, d = 1) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));
const kmToMi = (km) => km * 0.621371;

const state = {
  index: [], city: null, data: null,
  k: 5, mode: "clusters", featureKey: null,
  disabled: new Set(), log: true, varTarget: 0.86, seed: 42,
  result: null, names: {}, focusCluster: null, hoverIdx: null,
};
let map, mapReady = false, geojson = null;

// ---------------------------------------------------------------------------
// URL state
// ---------------------------------------------------------------------------
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("city")) state.city = p.get("city");
  if (p.get("k")) state.k = Math.max(2, Math.min(12, +p.get("k")));
  if (p.get("mode")) state.mode = p.get("mode");
  if (p.get("f")) state.featureKey = p.get("f");
  if (p.get("off")) state.disabled = new Set(p.get("off").split(",").filter(Boolean));
  if (p.get("log")) state.log = p.get("log") === "1";
  if (p.get("var")) state.varTarget = +p.get("var") / 100;
  if (p.get("seed")) state.seed = +p.get("seed");
}
function writeHash() {
  const p = new URLSearchParams();
  p.set("city", state.city); p.set("k", state.k); p.set("mode", state.mode);
  if (state.mode === "feature" && state.featureKey) p.set("f", state.featureKey);
  if (state.disabled.size) p.set("off", [...state.disabled].join(","));
  if (!state.log) p.set("log", "0");
  if (Math.round(state.varTarget * 100) !== 86) p.set("var", Math.round(state.varTarget * 100));
  if (state.seed !== 42) p.set("seed", state.seed);
  history.replaceState(null, "", "#" + p.toString());
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------
async function loadIndex() {
  state.index = await (await fetch("data/index.json")).json();
  const sel = $("#city-select");
  sel.innerHTML = "";
  for (const c of state.index) {
    const o = el("option"); o.value = c.slug; o.textContent = `${c.name}${c.country ? ", " + c.country : ""}`; sel.appendChild(o);
  }
  if (!state.city || !state.index.some(c => c.slug === state.city)) state.city = state.index[0].slug;
  sel.value = state.city;
}

async function loadCity(slug) {
  $("#summary-text").textContent = "Loading " + slug + "…";
  state.city = slug;
  state.data = await (await fetch(`data/${slug}.json`)).json();
  state.names = {};
  state.focusCluster = null;
  // Drop disabled keys that don't exist for this city; keep the user's choice otherwise.
  const keys = new Set(state.data.features.map(f => f.key));
  if (!state.featureKey || !keys.has(state.featureKey)) state.featureKey = state.data.features[0].key;
  buildGroupsUI();
  buildFeatureSelect();
  buildGeojson();
  if (mapReady) { setMapData(); fitToCity(); }
  compute();
}

// ---------------------------------------------------------------------------
// UI construction
// ---------------------------------------------------------------------------
function buildGroupsUI() {
  const root = $("#groups"); root.innerHTML = "";
  const byGroup = {};
  for (const f of state.data.features) (byGroup[f.group] ||= []).push(f);
  for (const g of state.data.groups) {
    const feats = byGroup[g]; if (!feats) continue;
    const det = el("details", "group");
    const sum = el("summary");
    const cb = el("input"); cb.type = "checkbox";
    const on = feats.filter(f => !state.disabled.has(f.key)).length;
    cb.checked = on === feats.length; cb.indeterminate = on > 0 && on < feats.length;
    cb.addEventListener("click", (e) => { e.stopPropagation(); });
    cb.addEventListener("change", () => {
      for (const f of feats) { if (cb.checked) state.disabled.delete(f.key); else state.disabled.add(f.key); }
      buildGroupsUI(); compute();
    });
    sum.append(cb, el("span", "gname", g), el("span", "gcount", `${on}/${feats.length}`), el("span", "caret", "▶"));
    det.appendChild(sum);
    const items = el("div", "items");
    for (const f of feats) {
      const lab = el("label");
      const c = el("input"); c.type = "checkbox"; c.checked = !state.disabled.has(f.key);
      c.addEventListener("change", () => { if (c.checked) state.disabled.delete(f.key); else state.disabled.add(f.key); buildGroupsUI(); compute(); });
      lab.append(c, document.createTextNode(f.label));
      items.appendChild(lab);
    }
    det.appendChild(items);
    root.appendChild(det);
  }
  const total = state.data.features.length, enabled = state.data.features.filter(f => !state.disabled.has(f.key)).length;
  $("#feature-count").textContent = `${enabled} of ${total} on`;
}

function buildFeatureSelect() {
  const sel = $("#feature-select"); sel.innerHTML = "";
  let og = null, last = null;
  for (const f of state.data.features) {
    if (f.group !== last) { og = el("optgroup"); og.label = f.group; sel.appendChild(og); last = f.group; }
    const o = el("option"); o.value = f.key; o.textContent = f.label; og.appendChild(o);
  }
  sel.value = state.featureKey;
}

function buildGeojson() {
  geojson = {
    type: "FeatureCollection",
    features: state.data.cells.map((c, i) => ({
      type: "Feature", id: i,
      properties: { idx: i, color: "#555" },
      geometry: { type: "Polygon", coordinates: [[...c.p, c.p[0]]] },
    })),
  };
}

// ---------------------------------------------------------------------------
// Compute: vectors -> PCA -> k-means
// ---------------------------------------------------------------------------
function compute() {
  const d = state.data; if (!d) return;
  const feats = d.features.map((f, j) => ({ ...f, j })).filter(f => !state.disabled.has(f.key));
  if (feats.length < 2) { $("#summary-text").textContent = "Select at least two features."; return; }
  const X = d.cells.map(c => feats.map(f => (state.log && f.kind === "density") ? Math.log1p(c.v[f.j]) : c.v[f.j]));
  const st = standardize(X);
  const usedFeats = st.keep.map(j => feats[j]);
  const p = pca(st.Z, state.varTarget);
  const km = kmeans(p.scores, state.k, { seed: state.seed });

  // order clusters by mean distance to centre (1 = most central), like the article's list
  const k = state.k;
  const agg = Array.from({ length: k }, () => ({ n: 0, dist: 0, land: 0, z: new Float64Array(usedFeats.length) }));
  km.labels.forEach((l, i) => {
    const a = agg[l]; a.n++; a.dist += d.cells[i].dist_km; a.land += d.cells[i].land_km2;
    const z = st.Z[i]; for (let j = 0; j < z.length; j++) a.z[j] += z[j];
  });
  const order = agg.map((a, i) => i).filter(i => agg[i].n > 0).sort((a, b) => agg[a].dist / agg[a].n - agg[b].dist / agg[b].n);
  const remap = new Int32Array(k).fill(-1); order.forEach((old, ni) => { remap[old] = ni; });
  const labels = km.labels.map(l => remap[l]);
  const clusters = order.map((old, ni) => {
    const a = agg[old];
    const z = Array.from(a.z, v => v / a.n);
    const ranked = z.map((v, j) => ({ f: usedFeats[j], z: v })).sort((x, y) => y.z - x.z);
    return {
      id: ni, color: PALETTE[ni % PALETTE.length], n: a.n, land: a.land, dist: a.dist / a.n, z,
      hi: ranked.filter(r => r.z > 0.25).slice(0, 4), lo: ranked.slice().reverse().filter(r => r.z < -0.25).slice(0, 3),
    };
  });
  autoName(clusters, usedFeats);

  // correlations of first two PCs with the original (transformed) features
  const pcs = [0, 1].filter(c => c < p.nComponents).map(c => {
    const s = p.scores.map(r => r[c]);
    const cors = usedFeats.map((f, j) => ({ f, r: pearson(s, st.Z.map(r => r[j])) })).sort((a, b) => b.r - a.r);
    return { c, ratio: p.ratios[c], pos: cors.slice(0, 5), neg: cors.slice(-5).reverse(), scores: s };
  });

  state.result = { feats: usedFeats, Z: st.Z, pca: p, labels, clusters, pcs, X };
  renderSummary(); renderClusters(); renderPCs(); renderScatter(); paintMap(); writeHash();
}

// Pick evocative names from archetypes based on each cluster's centroid z-scores.
function autoName(clusters, feats) {
  const idx = Object.fromEntries(feats.map((f, j) => [f.key, j]));
  const g = (c, keys) => { let s = 0, n = 0; for (const k of keys) if (k in idx) { s += c.z[idx[k]]; n++; } return n ? s / n : 0; };
  const grp = (c, name) => { const js = feats.map((f, j) => f.group === name ? j : -1).filter(j => j >= 0); return js.length ? js.reduce((s, j) => s + c.z[j], 0) / js.length : 0; };
  const archetypes = [
    ["The Core", c => grp(c, "Nightlife") + grp(c, "Food & Drink") + grp(c, "Culture") + g(c, ["tall_buildings", "offices", "hotels"]) * 1.5],
    ["The Ring", c => (grp(c, "Food & Drink") + grp(c, "Retail") + g(c, ["apartments", "subway_entrances", "rail_stations"])) * 0.9 - g(c, ["tall_buildings", "hotels", "attractions"]) * 0.8],
    ["The Meadows", c => g(c, ["green_frac", "houses", "trees", "parks"]) * 1.5 - grp(c, "Nightlife") - grp(c, "Retail")],
    ["The Bends", c => g(c, ["residential_frac", "schools", "places_of_worship", "grocery", "bus_stops"]) - g(c, ["tall_buildings", "offices", "hotels", "green_frac"]) * 0.7],
    ["The Works", c => g(c, ["industrial_frac", "service_roads", "car_parking", "fuel_charging"]) * 1.4 - g(c, ["residential_frac"])],
    ["The Shore", c => g(c, ["water_frac", "beach_frac"]) * 2 - g(c, ["buildings"]) * 0.5],
    ["The Commons", c => g(c, ["higher_ed", "healthcare", "libraries", "community", "sports"]) * 1.4],
    ["The Strip", c => g(c, ["major_roads", "fast_food", "car_parking", "shops_total", "fuel_charging"]) * 1.2 - g(c, ["pedestrian_paths"]) * 0.5],
    ["The Quiet Grid", c => g(c, ["local_streets", "houses", "residential_frac"]) - grp(c, "Food & Drink") - grp(c, "Nightlife")],
    ["The Edge", c => -grp(c, "Food & Drink") - grp(c, "Retail") - grp(c, "Civic") - g(c, ["buildings"])],
  ];
  const scores = [];
  for (const c of clusters) for (const [name, fn] of archetypes) scores.push({ c, name, s: fn(c) });
  scores.sort((a, b) => b.s - a.s);
  const used = new Set(), named = new Set();
  for (const { c, name } of scores) {
    if (used.has(name) || named.has(c.id)) continue;
    c.autoName = name; used.add(name); named.add(c.id);
  }
  for (const c of clusters) {
    if (!c.autoName) c.autoName = `Borough ${c.id + 1}`;
    c.name = state.names[c.id] || c.autoName;
  }
}

// ---------------------------------------------------------------------------
// Rendering: panel
// ---------------------------------------------------------------------------
function renderSummary() {
  const d = state.data, r = state.result, p = r.pca;
  const totalLand = d.cells.reduce((s, c) => s + c.land_km2, 0);
  $("#summary-text").innerHTML =
    `<b>${d.name}</b> · ${d.cells.length} hex neighborhoods (r = ${d.hex_radius_m} m, ${fmt(totalLand, 0)} km² of land) · ` +
    `<b>${r.feats.length}D</b> feature space → <b>${p.nComponents}D</b> after PCA (${(p.explained * 100).toFixed(0)}% of variance) → ` +
    `<b>K = ${state.k}</b> boroughs. Centre: ${d.center_label}. OSM snapshot ${d.generated}.`;
}

function renderClusters() {
  const root = $("#clusters"); root.innerHTML = "";
  const r = state.result;
  const totalN = r.labels.length;
  $("#clusters-title").textContent = `The new ${r.clusters.length} boroughs`;
  for (const c of r.clusters) {
    const card = el("div", "cluster" + (state.focusCluster != null && state.focusCluster !== c.id ? " dim" : ""));
    const sw = el("div", "swatch"); sw.style.background = c.color;
    const body = el("div");
    const name = el("div", "name", `${c.id + 1}. ${c.name}`);
    name.contentEditable = "true"; name.spellcheck = false; name.title = "Click to rename";
    name.addEventListener("blur", () => {
      const t = name.textContent.replace(/^\d+\.\s*/, "").trim();
      if (t && t !== c.autoName) state.names[c.id] = t; else delete state.names[c.id];
      c.name = t || c.autoName; renderLegend();
    });
    name.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); name.blur(); } });
    const dens = r.feats.length;
    const stats = el("div", "stats",
      `${c.n} cells (${(100 * c.n / totalN).toFixed(0)}%) · ${fmt(c.land, 0)} km² · avg ${fmt(c.dist)} km (${fmt(kmToMi(c.dist))} mi) from centre`);
    const tags = el("div", "tags");
    for (const h of c.hi) tags.appendChild(el("span", "tag hi", `▲ ${h.f.label} <b>${h.z >= 0 ? "+" : ""}${h.z.toFixed(1)}σ</b>`));
    for (const l of c.lo) tags.appendChild(el("span", "tag lo", `▼ ${l.f.label} <b>${l.z.toFixed(1)}σ</b>`));
    if (!c.hi.length && !c.lo.length) tags.appendChild(el("span", "tag", "close to the city average"));
    body.append(name, stats, tags);
    card.append(sw, body);
    card.addEventListener("click", (e) => {
      if (e.target === name) return;
      state.focusCluster = state.focusCluster === c.id ? null : c.id;
      renderClusters(); paintMap(); renderScatter();
    });
    root.appendChild(card);
  }
  renderLegend();
}

function renderPCs() {
  const root = $("#pcs"); root.innerHTML = "";
  for (const pc of state.result.pcs) {
    const box = el("div", "pc");
    box.appendChild(el("h3", null, `d${pc.c + 1} — explains ${(pc.ratio * 100).toFixed(0)}% of variance`));
    const bars = el("div", "bars");
    const col = (title, list, cls) => {
      const d = el("div"); d.appendChild(el("b", null, title));
      for (const { f, r } of list) {
        const row = el("div", "corr");
        const bar = el("span", "bar " + cls); bar.style.width = `${Math.abs(r) * 60}px`;
        row.append(bar, document.createTextNode(`${f.label} (${r >= 0 ? "+" : ""}${r.toFixed(2)})`));
        d.appendChild(row);
      }
      return d;
    };
    bars.append(col("Top positive correlates", pc.pos, "pos"), col("Top negative correlates", pc.neg, "neg"));
    box.appendChild(bars);
    root.appendChild(box);
  }
}

function renderScatter() {
  const cv = $("#scatter"), ctx = cv.getContext("2d"), r = state.result;
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#12151b"; ctx.fillRect(0, 0, W, H);
  if (!r || r.pcs.length < 2) { ctx.fillStyle = "#9aa3b2"; ctx.font = "12px sans-serif"; ctx.fillText("Need ≥ 2 components", 12, 20); return; }
  const xs = r.pcs[0].scores, ys = r.pcs[1].scores;
  const sx = xs.slice().sort((a, b) => a - b), sy = ys.slice().sort((a, b) => a - b);
  const x0 = quantile(sx, 0.005), x1 = quantile(sx, 0.995), y0 = quantile(sy, 0.005), y1 = quantile(sy, 0.995);
  const pad = 14;
  const X = v => pad + (Math.min(Math.max(v, x0), x1) - x0) / (x1 - x0 || 1) * (W - 2 * pad);
  const Y = v => H - pad - (Math.min(Math.max(v, y0), y1) - y0) / (y1 - y0 || 1) * (H - 2 * pad);
  ctx.strokeStyle = "#2a2f3a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), pad); ctx.lineTo(X(0), H - pad); ctx.moveTo(pad, Y(0)); ctx.lineTo(W - pad, Y(0)); ctx.stroke();
  ctx.fillStyle = "#9aa3b2"; ctx.font = "10px sans-serif";
  ctx.fillText("d1 →", W - pad - 26, Y(0) - 4); ctx.fillText("d2 ↑", X(0) + 4, pad + 8);
  for (let i = 0; i < xs.length; i++) {
    const c = r.clusters[r.labels[i]];
    const dim = state.focusCluster != null && state.focusCluster !== c.id;
    ctx.globalAlpha = dim ? 0.12 : 0.8;
    ctx.fillStyle = c.color;
    ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), 2.4, 0, Math.PI * 2); ctx.fill();
  }
  if (state.hoverIdx != null) {
    const i = state.hoverIdx;
    ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), 5, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------
function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
    center: [-74, 40.7], zoom: 9.5, attributionControl: false,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: "© OpenStreetMap contributors" }));
  map.on("load", () => {
    map.addSource("cells", { type: "geojson", data: geojson || { type: "FeatureCollection", features: [] } });
    map.addSource("boundary", { type: "geojson", data: state.data ? { type: "Feature", geometry: state.data.boundary } : { type: "FeatureCollection", features: [] } });
    map.addSource("labels", { type: "raster", tiles: ["https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}@2x.png"], tileSize: 256 });
    map.addLayer({
      id: "cells-fill", type: "fill", source: "cells",
      paint: { "fill-color": ["get", "color"], "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.95, ["get", "opacity"]] },
    });
    map.addLayer({ id: "cells-line", type: "line", source: "cells", paint: { "line-color": "#0f1115", "line-width": 0.6, "line-opacity": 0.6 } });
    map.addLayer({ id: "cells-hover", type: "line", source: "cells", paint: { "line-color": "#ffffff", "line-width": ["case", ["boolean", ["feature-state", "hover"], false], 2, 0] } });
    map.addLayer({ id: "boundary", type: "line", source: "boundary", paint: { "line-color": "#ffffff", "line-width": 1, "line-opacity": 0.35 } });
    map.addLayer({ id: "labels", type: "raster", source: "labels", paint: { "raster-opacity": 0.9 } });
    mapReady = true;
    if (state.data) { setMapData(); fitToCity(); paintMap(); }

    let hovered = null;
    const tip = $("#tooltip");
    map.on("mousemove", "cells-fill", (e) => {
      const f = e.features[0]; if (!f) return;
      if (hovered !== null && hovered !== f.id) map.setFeatureState({ source: "cells", id: hovered }, { hover: false });
      hovered = f.id; map.setFeatureState({ source: "cells", id: hovered }, { hover: true });
      map.getCanvas().style.cursor = "pointer";
      state.hoverIdx = f.properties.idx;
      tip.innerHTML = tooltipHtml(f.properties.idx);
      tip.hidden = false;
      const wrap = $("#map-wrap").getBoundingClientRect();
      let x = e.point.x + 14, y = e.point.y + 14;
      if (x + 270 > wrap.width) x = e.point.x - 270; if (y + 200 > wrap.height) y = e.point.y - 180;
      tip.style.left = x + "px"; tip.style.top = y + "px";
      renderScatter();
    });
    map.on("mouseleave", "cells-fill", () => {
      if (hovered !== null) map.setFeatureState({ source: "cells", id: hovered }, { hover: false });
      hovered = null; state.hoverIdx = null; tip.hidden = true; map.getCanvas().style.cursor = ""; renderScatter();
    });
    map.on("click", "cells-fill", (e) => {
      const f = e.features[0]; if (!f || !state.result) return;
      const cl = state.result.labels[f.properties.idx];
      state.focusCluster = state.focusCluster === cl ? null : cl;
      renderClusters(); paintMap(); renderScatter();
    });
  });
}

function setMapData() {
  map.getSource("cells").setData(geojson);
  map.getSource("boundary").setData({ type: "Feature", geometry: state.data.boundary });
}

function fitToCity() {
  // ignore far-flung outliers (e.g. San Francisco's Farallon Islands) when framing the city
  const cells = state.data.cells;
  const lons = cells.map(c => c.c[0]).sort((a, b) => a - b), lats = cells.map(c => c.c[1]).sort((a, b) => a - b);
  const mlon = quantile(lons, 0.5), mlat = quantile(lats, 0.5), k = Math.cos(mlat * Math.PI / 180);
  const dist = cells.map(c => Math.hypot((c.c[0] - mlon) * k, c.c[1] - mlat));
  const cutoff = 2.5 * quantile(dist.slice().sort((a, b) => a - b), 0.95);
  const b = new maplibregl.LngLatBounds();
  cells.forEach((c, i) => { if (dist[i] <= cutoff) b.extend(c.c); });
  map.fitBounds(b, { padding: { top: 30, right: 40, bottom: 40, left: 60 }, duration: 800 });
}

function ramp(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1), i = Math.floor(x), f = x - i;
  if (i >= VIRIDIS.length - 1) return VIRIDIS[VIRIDIS.length - 1];
  const a = VIRIDIS[i].match(/\w\w/g).map(h => parseInt(h, 16)), b = VIRIDIS[i + 1].match(/\w\w/g).map(h => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}

function continuousValues() {
  const r = state.result, d = state.data;
  if (state.mode === "feature") {
    const j = d.features.findIndex(f => f.key === state.featureKey);
    return { values: d.cells.map(c => c.v[j]), label: d.features[j].label, unit: d.features[j].kind === "density" ? "per km²" : "fraction of land" };
  }
  const c = state.mode === "pc1" ? 1 : 0;
  if (!r.pcs[c]) return null;
  return { values: r.pcs[c].scores, label: `d${c + 1} (PCA component ${c + 1})`, unit: "score" };
}

function paintMap() {
  if (!mapReady || !state.result || !geojson) return;
  const r = state.result;
  let legend;
  if (state.mode === "clusters") {
    geojson.features.forEach((f, i) => {
      const cl = r.clusters[r.labels[i]];
      f.properties.color = cl.color;
      f.properties.opacity = state.focusCluster != null && state.focusCluster !== cl.id ? 0.08 : 0.72;
    });
  } else {
    const cv = continuousValues();
    if (!cv) return;
    const sorted = cv.values.slice().sort((a, b) => a - b);
    const lo = quantile(sorted, 0.02), hi = quantile(sorted, 0.98);
    // rank-based colouring is more legible for heavy-tailed densities
    const rank = new Map(); sorted.forEach((v, i) => { if (!rank.has(v)) rank.set(v, i / (sorted.length - 1 || 1)); });
    geojson.features.forEach((f, i) => { f.properties.color = ramp(rank.get(cv.values[i])); f.properties.opacity = 0.78; });
    legend = { type: "ramp", label: cv.label, unit: cv.unit, lo, hi };
  }
  map.getSource("cells").setData(geojson);
  renderLegend(legend);
}

function renderLegend(cont) {
  const lg = $("#legend"); lg.hidden = false; lg.innerHTML = "";
  if (state.mode === "clusters") {
    lg.appendChild(el("div", null, `<b>${state.data.name}</b> · K = ${state.k}`));
    for (const c of state.result.clusters) {
      const it = el("div", "item"); const i = el("i"); i.style.background = c.color;
      it.append(i, document.createTextNode(`${c.id + 1}. ${c.name}`)); lg.appendChild(it);
    }
    lg.appendChild(el("div", "muted small", "click a borough to isolate it"));
  } else if (cont) {
    lg.appendChild(el("div", null, `<b>${cont.label}</b>`));
    const rp = el("div", "ramp"); rp.style.background = `linear-gradient(90deg, ${VIRIDIS.join(",")})`;
    lg.appendChild(rp);
    lg.appendChild(el("div", "ends", `<span>${fmt(cont.lo, 2)}</span><span>${cont.unit} (rank-scaled)</span><span>${fmt(cont.hi, 2)}</span>`));
  }
}

function tooltipHtml(i) {
  const d = state.data, r = state.result, c = d.cells[i];
  const cl = r.clusters[r.labels[i]];
  const rows = [];
  rows.push(`<div class="r"><span>Distance to ${d.center_label}</span><b>${fmt(c.dist_km)} km / ${fmt(kmToMi(c.dist_km))} mi</b></div>`);
  if (r.pcs[0]) rows.push(`<div class="r"><span>d1 / d2</span><b>${r.pcs[0].scores[i].toFixed(2)} / ${r.pcs[1] ? r.pcs[1].scores[i].toFixed(2) : "–"}</b></div>`);
  // most distinctive features of this cell (largest |z|)
  const z = r.Z[i].map((v, j) => ({ f: r.feats[j], v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 4);
  for (const { f, v } of z) {
    const raw = c.v[f.j];
    const val = f.kind === "fraction" ? `${(raw * 100).toFixed(0)}%` : `${fmt(raw)}/km²`;
    rows.push(`<div class="r"><span>${f.label}</span><b>${val} <span class="muted">(${v >= 0 ? "+" : ""}${v.toFixed(1)}σ)</span></b></div>`);
  }
  if (state.mode === "feature") {
    const j = d.features.findIndex(f => f.key === state.featureKey), f = d.features[j];
    rows.unshift(`<div class="r"><span>${f.label}</span><b>${f.kind === "fraction" ? (c.v[j] * 100).toFixed(0) + "%" : fmt(c.v[j]) + "/km²"}</b></div>`);
  }
  return `<div class="t"><span style="color:${cl.color}">■</span> ${cl.id + 1}. ${cl.name}</div>${rows.join("")}`;
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
function bindControls() {
  $("#city-select").addEventListener("change", (e) => loadCity(e.target.value));
  const kr = $("#k-range");
  kr.value = state.k; $("#k-value").textContent = state.k;
  kr.addEventListener("input", () => { state.k = +kr.value; $("#k-value").textContent = state.k; state.focusCluster = null; state.names = {}; compute(); });
  $("#mode-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.mode = b.dataset.mode;
    document.querySelectorAll("#mode-seg button").forEach(x => x.classList.toggle("active", x === b));
    $("#feature-ctl").hidden = state.mode !== "feature";
    paintMap(); writeHash();
  });
  document.querySelectorAll("#mode-seg button").forEach(x => x.classList.toggle("active", x.dataset.mode === state.mode));
  $("#feature-ctl").hidden = state.mode !== "feature";
  $("#feature-select").addEventListener("change", (e) => { state.featureKey = e.target.value; paintMap(); writeHash(); });
  const lt = $("#log-toggle"); lt.checked = state.log;
  lt.addEventListener("change", () => { state.log = lt.checked; compute(); });
  const vr = $("#var-range"); vr.value = Math.round(state.varTarget * 100); $("#var-value").textContent = vr.value + "%";
  vr.addEventListener("input", () => { state.varTarget = +vr.value / 100; $("#var-value").textContent = vr.value + "%"; compute(); });
  const sd = $("#seed"); sd.value = state.seed;
  sd.addEventListener("change", () => { state.seed = +sd.value || 0; compute(); });
  $("#about-btn").addEventListener("click", () => { $("#about").hidden = false; });
  $("#about-close").addEventListener("click", () => { $("#about").hidden = true; });
  $("#about").addEventListener("click", (e) => { if (e.target.id === "about") $("#about").hidden = true; });
}

(async function main() {
  readHash();
  bindControls();
  initMap();
  await loadIndex();
  await loadCity(state.city);
})();
