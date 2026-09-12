#!/usr/bin/env python3
"""
Build the neighborhood feature dataset for one city from OpenStreetMap.

    python pipeline/build_city.py nyc [--target-cells 800] [--skip buildings,trees]

Steps
  1. Fetch the city's administrative boundary polygon (Nominatim).
  2. Lay a hexagonal grid over it (our "neighborhoods"; the Topos article used ZIP codes).
  3. Pull OSM features via the Overpass API (compact CSV responses, with automatic
     bbox splitting when a query is too heavy for the server).
  4. Bucket everything into per-cell counts / densities / land-use fractions.
  5. Write data/<slug>.json for the web app (PCA + k-means run in the browser).
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import requests
from shapely.geometry import Polygon, MultiPolygon, shape, Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

sys.path.insert(0, os.path.dirname(__file__))
import features as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "pipeline", "cache")
OUT = os.path.join(ROOT, "data")
UA = "five-boroughs-osm/1.0 (research demo; github pages app)"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------
def fetch_boundary(city):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{city['slug']}_boundary.json")
    if os.path.exists(path):
        return json.load(open(path))
    url = "https://nominatim.openstreetmap.org/lookup"
    r = requests.get(url, params={"osm_ids": f"R{city['osm_relation']}", "format": "json",
                                  "polygon_geojson": 1}, headers={"User-Agent": UA}, timeout=120)
    r.raise_for_status()
    res = r.json()
    if not res:
        raise SystemExit(f"Nominatim returned nothing for relation {city['osm_relation']}")
    log("boundary:", res[0].get("display_name"))
    json.dump(res[0]["geojson"], open(path, "w"))
    time.sleep(1.1)
    return res[0]["geojson"]


# ---------------------------------------------------------------------------
# Local projection (equirectangular around the city centre; good enough at city scale)
# ---------------------------------------------------------------------------
class Proj:
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = 111320.0 * math.cos(math.radians(lat0))
        self.ky = 110540.0

    def fwd(self, lon, lat):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inv(self, x, y):
        return x / self.kx + self.lon0, y / self.ky + self.lat0

    def geom_fwd(self, g):
        from shapely.ops import transform
        return transform(lambda x, y, z=None: self.fwd(x, y), g)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Hex grid (pointy-top) in projected metres
# ---------------------------------------------------------------------------
def hex_polygon(cx, cy, r):
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 30)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return Polygon(pts)


def make_grid(boundary_m, r):
    minx, miny, maxx, maxy = boundary_m.bounds
    w = math.sqrt(3) * r
    h = 1.5 * r
    prepared = prep(boundary_m)
    cells = []
    row = 0
    y = miny - r
    while y <= maxy + r:
        xoff = (w / 2) if (row % 2) else 0
        x = minx - w + xoff
        col = 0
        while x <= maxx + w:
            hp = hex_polygon(x, y, r)
            if prepared.intersects(hp):
                cells.append((row, col, hp))
            x += w
            col += 1
        y += h
        row += 1
    return cells


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------
class Overpass:
    def __init__(self, area_id):
        self.area_id = area_id
        self.ep = 0

    def _post(self, query, timeout):
        """POST a query. Raises RuntimeError quickly when the query is too heavy
        (so the caller can split the bbox); retries on rate limiting / flaky endpoints."""
        last = None
        for attempt in range(6):
            url = OVERPASS_ENDPOINTS[self.ep % len(OVERPASS_ENDPOINTS)]
            try:
                r = requests.post(url, data={"data": query}, headers={"User-Agent": UA}, timeout=timeout + 60)
            except requests.Timeout:
                raise RuntimeError("client timeout")
            except requests.ConnectionError as e:
                last = repr(e)
                self.ep += 1
                time.sleep(3)
                continue
            if r.status_code == 200:
                tail = r.text[-3000:]
                if "runtime error" in tail or "Query timed out" in tail or "out of memory" in tail:
                    raise RuntimeError("overpass runtime error: " + tail[-160:].strip())
                return r.text
            last = f"HTTP {r.status_code}: {r.text[:160]}"
            if r.status_code == 504:
                if "too busy" in r.text or "Dispatcher_Client" in r.text:
                    # dispatcher refused to schedule us (load / memory); wait and retry, try another endpoint
                    log(f"  overpass busy ({url.split('/')[2]}), retrying in 20s")
                    self.ep += 1
                    time.sleep(20)
                    continue
                raise RuntimeError(last)
            if r.status_code in (429, 502, 503):
                log(f"  HTTP {r.status_code} from {url.split('/')[2]}, retrying")
                self.ep += 1
                time.sleep(15 if r.status_code == 429 else 5)
                continue
            raise RuntimeError(last)
        raise RuntimeError(f"overpass failed: {last}")

    def csv(self, body, cols, bbox=None, timeout=240, depth=0):
        """Return list of dict rows with lat, lon + cols. Splits bbox on failure/oversize."""
        bb = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})" if bbox else ""
        colspec = ",".join(["::lat", "::lon"] + [f'"{c}"' if ":" in c else c for c in cols])
        q = (f"[out:csv({colspec};false)][timeout:{timeout}][maxsize:268435456];"
             f"area(id:{self.area_id})->.a;(" + body.replace("{bbox}", bb) + ");out center;")
        try:
            txt = self._post(q, timeout)
        except RuntimeError as e:
            if depth >= 4 or bbox is None and depth > 0:
                raise
            log(f"  split (depth {depth}) after: {str(e)[:80]}")
            return self._split(body, cols, bbox, timeout, depth)
        rows = []
        rdr = csv.reader(io.StringIO(txt), delimiter="\t")
        for rec in rdr:
            if len(rec) < 2 or not rec[0]:
                continue
            try:
                d = {"lat": float(rec[0]), "lon": float(rec[1])}
            except ValueError:
                continue
            for i, c in enumerate(cols):
                d[c] = rec[2 + i] if 2 + i < len(rec) else ""
            rows.append(d)
        return rows

    def _split(self, body, cols, bbox, timeout, depth):
        s, w, n, e = bbox
        mlat, mlon = (s + n) / 2, (w + e) / 2
        rows = []
        for sub in [(s, w, mlat, mlon), (s, mlon, mlat, e), (mlat, w, n, mlon), (mlat, mlon, n, e)]:
            time.sleep(2)
            rows += self.csv(body, cols, sub, timeout, depth + 1)
        return rows

    def json_geom(self, body, bbox, timeout=300, depth=0, area=True):
        """JSON 'out geom' query. If area=False the body must rely on {bbox} only."""
        bb = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
        pre = f"area(id:{self.area_id})->.a;" if area else ""
        q = (f"[out:json][timeout:{timeout}][maxsize:268435456];{pre}("
             + body.replace("{bbox}", bb) + ");out geom;")
        try:
            txt = self._post(q, timeout)
            return json.loads(txt)["elements"]
        except (RuntimeError, ValueError) as e:
            if depth >= 3:
                raise
            log(f"  split geom (depth {depth}) after: {str(e)[:80]}")
            s, w, n, e_ = bbox
            mlat, mlon = (s + n) / 2, (w + e_) / 2
            els = []
            for sub in [(s, w, mlat, mlon), (s, mlon, mlat, e_), (mlat, w, n, mlon), (mlat, mlon, n, e_)]:
                time.sleep(2)
                els += self.json_geom(body, sub, timeout, depth + 1, area)
            return els


def cached(slug, name, fn):
    path = os.path.join(CACHE, f"{slug}_{name}.json")
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time()
    data = fn()
    json.dump(data, open(path, "w"))
    log(f"  {name}: {len(data)} rows in {time.time() - t:.0f}s")
    return data


# ---------------------------------------------------------------------------
# Land-use polygons from Overpass 'out geom' elements
# ---------------------------------------------------------------------------
def stitch_rings(ways):
    """ways: list of coordinate lists. Join by shared endpoints into closed rings."""
    rings, pool = [], [list(w) for w in ways if len(w) >= 2]
    while pool:
        cur = pool.pop()
        changed = True
        while changed and cur[0] != cur[-1]:
            changed = False
            for i, w in enumerate(pool):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[::-1][1:]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                pool.pop(i)
                changed = True
                break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def elements_to_polygons(elements):
    """Yield (tags, shapely polygon in lon/lat)."""
    for el in elements:
        tags = el.get("tags", {})
        polys = []
        if el["type"] == "way" and "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                polys.append(coords)
        elif el["type"] == "relation":
            outers = [[(p["lon"], p["lat"]) for p in m.get("geometry", [])]
                      for m in el.get("members", []) if m.get("type") == "way" and m.get("role") in ("outer", "")]
            inners = [[(p["lon"], p["lat"]) for p in m.get("geometry", [])]
                      for m in el.get("members", []) if m.get("type") == "way" and m.get("role") == "inner"]
            outer_rings = stitch_rings(outers)
            inner_rings = stitch_rings(inners)
            for o in outer_rings:
                try:
                    p = Polygon(o, [i for i in inner_rings if Polygon(o).contains(Point(i[0]))])
                    if not p.is_valid:
                        p = p.buffer(0)
                    if not p.is_empty:
                        yield tags, p
                except Exception:
                    pass
            continue
        for c in polys:
            try:
                p = Polygon(c)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    yield tags, p
            except Exception:
                pass


def coastal_water_mask(ov, slug, boundary_m, proj, bbox, building_pts_m, sample_step=1):
    """
    Seas, bays and tidal rivers are not water polygons in OSM; they are implied by
    natural=coastline lines. Split the boundary polygon along all coastline ways and
    call every resulting piece with (almost) no buildings 'water'. Returns a
    (Multi)Polygon of water in projected metres, or None.
    """
    from shapely.geometry import LineString
    from shapely.ops import polygonize

    els = cached(slug, "coastline", lambda: ov.json_geom('way["natural"="coastline"]{bbox};', bbox, area=False))
    lines = []
    for el in els:
        if el.get("type") == "way" and "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) >= 2:
                lines.append(proj.geom_fwd(LineString(coords)))
    if not lines:
        return None
    lines = [l.intersection(boundary_m) for l in lines]
    lines = [l for l in lines if not l.is_empty]
    noded = unary_union(lines + [boundary_m.boundary])
    pieces = [p for p in polygonize(noded) if p.representative_point().within(boundary_m)]
    if len(pieces) <= 1:
        return None
    pts = building_pts_m
    tree = STRtree(pts) if len(pts) else None
    water = []
    for p in pieces:
        area_km2 = p.area / 1e6
        if area_km2 < 0.02:
            continue
        n = len(tree.query(p, predicate="contains")) * sample_step if tree is not None else 0
        if n / area_km2 < 15:  # fewer than 15 buildings per km2 -> open water
            water.append(p)
    log(f"coastline mask: {len(pieces)} pieces, {len(water)} water ({sum(w.area for w in water)/1e6:.0f} km2)")
    return unary_union(water) if water else None


def classify_landuse(tags):
    for cls, rules in F.LANDUSE_CLASSES.items():
        for key, vals in rules:
            if tags.get(key) in vals:
                return cls
    return None


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(city, target_cells=800, skip=()):
    slug = city["slug"]
    log(f"=== {city['name']} ({slug}) ===")
    bnd_geo = fetch_boundary(city)
    boundary = shape(bnd_geo)
    if not boundary.is_valid:
        boundary = boundary.buffer(0)
    lon0, lat0 = boundary.centroid.x, boundary.centroid.y
    proj = Proj(lon0, lat0)
    boundary_m = proj.geom_fwd(boundary)
    area_km2 = boundary_m.area / 1e6
    s, w, n, e = boundary.bounds[1], boundary.bounds[0], boundary.bounds[3], boundary.bounds[2]
    bbox = (s, w, n, e)
    log(f"boundary area {area_km2:.0f} km2, bbox {bbox}")

    ov = Overpass(3600000000 + city["osm_relation"])

    # --- land-use polygons (tiled proactively: 'out geom' responses are big) -----
    def fetch_landuse():
        nt = 1 if area_km2 < 300 else (2 if area_km2 < 900 else 3)
        seen, els = set(), []
        for i in range(nt):
            for j in range(nt):
                sub = (s + (n - s) * i / nt, w + (e - w) * j / nt, s + (n - s) * (i + 1) / nt, w + (e - w) * (j + 1) / nt)
                t0 = time.time()
                tile = ov.json_geom(F.LANDUSE_QUERY, sub, timeout=240)
                for el in tile:
                    key = (el["type"], el["id"])
                    if key not in seen:
                        seen.add(key)
                        els.append(el)
                log(f"  landuse tile {i * nt + j + 1}/{nt * nt}: {len(tile)} elements in {time.time() - t0:.0f}s")
                time.sleep(1)
        return els
    lu_elements = cached(slug, "landuse", fetch_landuse)
    lu_polys = []  # (class, geom_m)
    for tags, p in elements_to_polygons(lu_elements):
        cls = classify_landuse(tags)
        if cls:
            lu_polys.append((cls, proj.geom_fwd(p)))
    log(f"landuse polygons: {len(lu_polys)}")
    lu_geoms = [g for _, g in lu_polys]
    lu_tree = STRtree(lu_geoms) if lu_geoms else None

    # --- point sources ------------------------------------------------------
    point_rows = {}
    for name, spec in F.POINT_SOURCES.items():
        if name in skip:
            log(f"skipping source {name}")
            point_rows[name] = []
            continue
        point_rows[name] = cached(slug, name, lambda spec=spec: ov.csv(spec["query"], spec["cols"], bbox))

    # --- coastal water mask (sea / bays / tidal rivers) ---------------------------
    land_m = boundary_m
    if "coastline" not in skip:
        bld = point_rows.get("buildings") or point_rows.get("streets") or []
        step = max(1, len(bld) // 150000)
        pts = [Point(*proj.fwd(rw["lon"], rw["lat"])) for rw in bld[::step]]
        try:
            water_m = coastal_water_mask(ov, slug, boundary_m, proj, bbox, pts, step)
        except Exception as ex:
            log(f"coastline mask failed ({ex}); continuing without it")
            water_m = None
        if water_m is not None:
            land_m = boundary_m.difference(water_m)
            log(f"land after coastal mask: {land_m.area/1e6:.0f} km2 (boundary {boundary_m.area/1e6:.0f} km2)")

    # --- hex grid: size chosen to hit ~target_cells cells over the land area ------
    r = math.sqrt(land_m.area / (target_cells * 2.598))
    r = float(min(1200, max(250, r)))
    cells = make_grid(boundary_m, r)
    log(f"hex radius {r:.0f} m -> {len(cells)} candidate cells")

    # --- assign points to cells ------------------------------------------------
    hex_w = math.sqrt(3) * r
    cell_index = {}
    for i, (row, col, hp) in enumerate(cells):
        cell_index[(row, col)] = i
    minx, miny = boundary_m.bounds[0] - hex_w, boundary_m.bounds[1] - r
    # grid origin as in make_grid: rows at y = miny0 + row*1.5r, x = minx0 + col*w + (row odd ? w/2 : 0)
    miny0 = boundary_m.bounds[1] - r
    minx0 = boundary_m.bounds[0] - hex_w

    def locate(x, y):
        """Find hex cell (row, col) containing projected point using nearest-centre search."""
        row_f = (y - miny0) / (1.5 * r)
        best, bestd = None, 1e18
        for row in (int(math.floor(row_f)), int(math.floor(row_f)) + 1):
            xoff = (hex_w / 2) if (row % 2) else 0
            col_f = (x - minx0 - xoff) / hex_w
            for col in (int(math.floor(col_f)), int(math.floor(col_f)) + 1):
                cx = minx0 + col * hex_w + xoff
                cy = miny0 + row * 1.5 * r
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < bestd:
                    best, bestd = (row, col), d
        return cell_index.get(best)

    feat_keys = [k for k, *_ in F.POINT_FEATURES] + [k for k, *_ in F.AREA_FEATURES]
    counts = np.zeros((len(cells), len(F.POINT_FEATURES)), dtype=np.float64)
    pf_by_source = defaultdict(list)
    for j, (key, label, group, source, matcher) in enumerate(F.POINT_FEATURES):
        pf_by_source[source].append((j, matcher))
    for source, rows in point_rows.items():
        matchers = pf_by_source.get(source, [])
        for rrow in rows:
            x, y = proj.fwd(rrow["lon"], rrow["lat"])
            ci = locate(x, y)
            if ci is None:
                continue
            for j, m in matchers:
                if m(rrow):
                    counts[ci, j] += 1
    log("points assigned")

    # --- per-cell geometry, land area, land-use fractions ------------------------
    lu_cls = [c for c, _ in lu_polys]
    out_cells = []
    hex_area = 2.598076 * r * r
    kept = 0
    land_prep = prep(land_m)
    for i, (row, col, hp) in enumerate(cells):
        if not land_prep.intersects(hp):
            continue
        inter = hp.intersection(land_m)
        if inter.is_empty or inter.area < 0.15 * hex_area:
            continue
        cls_area = defaultdict(float)
        if lu_tree is not None:
            for idx in lu_tree.query(inter):
                g = lu_geoms[idx]
                try:
                    a = inter.intersection(g).area
                except Exception:
                    continue
                if a > 0:
                    cls_area[lu_cls[idx]] += a
        water = min(cls_area.get("water", 0.0), inter.area)
        land = max(inter.area - water, 1.0)
        land_km2 = land / 1e6
        if land < 0.12 * hex_area:
            continue
        c = counts[i]
        # water/uninhabited filter: need some streets or buildings or POIs
        streets = sum(c[j] for j, (k, *_r) in enumerate(F.POINT_FEATURES) if _r[1] == "Streets")
        blds = c[feat_keys.index("buildings")]
        pois = c.sum() - streets - blds - c[feat_keys.index("trees")]
        if streets < 3 and blds < 3 and pois < 2:
            continue
        dens = [round(float(v) / land_km2, 3) for v in c]
        fracs = []
        for key, label, group, cls in F.AREA_FEATURES:
            denom = inter.area if cls == "water" else land
            fracs.append(round(min(cls_area.get(cls, 0.0) / denom, 1.0), 4))
        cen = hp.centroid
        clon, clat = proj.inv(cen.x, cen.y)
        poly = [[round(v, 5) for v in proj.inv(px, py)] for px, py in list(hp.exterior.coords)[:-1]]
        out_cells.append({
            "id": f"{row}_{col}",
            "c": [round(clon, 5), round(clat, 5)],
            "p": poly,
            "land_km2": round(land_km2, 4),
            "dist_km": round(haversine_km(clon, clat, city["center"][0], city["center"][1]), 2),
            "v": dens + fracs,
            "n": [int(v) for v in c],
        })
        kept += 1
    log(f"kept {kept} cells")

    features_meta = ([{"key": k, "label": l, "group": g, "kind": "density"} for k, l, g, *_ in F.POINT_FEATURES]
                     + [{"key": k, "label": l, "group": g, "kind": "fraction"} for k, l, g, _ in F.AREA_FEATURES])
    # drop features that are empty for this city (e.g. no trams)
    vals = np.array([c["v"] for c in out_cells])
    keep_idx = [j for j in range(vals.shape[1]) if np.count_nonzero(vals[:, j]) >= max(5, 0.02 * len(out_cells))]
    dropped = [features_meta[j]["key"] for j in range(vals.shape[1]) if j not in keep_idx]
    if dropped:
        log("dropping sparse features:", dropped)
    for c in out_cells:
        c["v"] = [c["v"][j] for j in keep_idx]
        c["n"] = [c["n"][j] for j in keep_idx if j < len(c["n"])]
    features_meta = [features_meta[j] for j in keep_idx]

    # simplified boundary for display
    bnd_simple = boundary.simplify(0.0005, preserve_topology=True)
    out = {
        "slug": slug,
        "name": city["name"],
        "country": city.get("country", ""),
        "osm_relation": city["osm_relation"],
        "center": city["center"],
        "center_label": city.get("center_label", "city centre"),
        "hex_radius_m": round(r),
        "generated": time.strftime("%Y-%m-%d"),
        "groups": F.GROUPS,
        "features": features_meta,
        "boundary": json.loads(json.dumps(bnd_simple.__geo_interface__)),
        "cells": out_cells,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{slug}.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return out


def update_index():
    cities = json.load(open(os.path.join(ROOT, "pipeline", "cities.json")))
    idx = []
    for c in cities:
        p = os.path.join(OUT, f"{c['slug']}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            idx.append({"slug": c["slug"], "name": c["name"], "country": c.get("country", ""),
                        "center": c["center"], "cells": len(d["cells"]), "generated": d["generated"]})
    json.dump(idx, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    log(f"index: {[c['slug'] for c in idx]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*", help="city slugs from pipeline/cities.json (default: all)")
    ap.add_argument("--target-cells", type=int, default=800)
    ap.add_argument("--skip", default="", help="comma list of point sources to skip (e.g. buildings,trees)")
    args = ap.parse_args()
    cities = json.load(open(os.path.join(ROOT, "pipeline", "cities.json")))
    by = {c["slug"]: c for c in cities}
    slugs = args.slugs or list(by)
    skip = tuple(s for s in args.skip.split(",") if s)
    for s in slugs:
        try:
            build(by[s], args.target_cells, skip)
        except Exception as ex:
            log(f"!! {s} failed: {ex}")
    update_index()
