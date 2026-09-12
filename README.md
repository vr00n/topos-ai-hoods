# Boroughs for the 21st Century — from OpenStreetMap, for any city

A recreation of Topos' 2017 essay
[*Five Boroughs for the 21st Century*](https://medium.com/topos-ai/five-boroughs-for-the-21st-century-8da941f53618),
rebuilt on open data and generalised to many cities.

Topos described every NYC neighborhood as a 65-dimensional vector, reduced it
with PCA to 16 dimensions (86 % of variance), and re-partitioned the city with
k-means (K = 5). The result was five concentric "boroughs" that ignored the
rivers: *Minhattan, The Ring, North Bend, South Bend, The Meadows*.

This project does the same thing with OpenStreetMap features and lets you turn
the knobs in the browser: pick a city, change K, switch feature groups on and
off, and inspect the principal components.

## How it works

```
pipeline/build_city.py  (offline, Python)          web app (static, runs in browser)
─────────────────────────────────────────          ────────────────────────────────────
Nominatim ─► city boundary polygon                 data/<city>.json
Overpass  ─► POIs, streets, buildings, trees   ─►  ├─ optional log1p on densities
             land-use polygons, coastline          ├─ z-score
hex grid  ─► ~800 "neighborhoods" per city         ├─ PCA (keep ≥ 86 % variance)
per-hex densities + land-use fractions             ├─ k-means++ (12 restarts)
                                                   └─ MapLibre choropleth + cluster cards
```

**Neighborhoods.** Instead of ZIP codes, the city's administrative boundary is
tiled with hexagons whose radius adapts to city size (≈ 800 land cells per
city). Seas, bays and tidal rivers are removed using `natural=coastline`;
inland lakes and rivers via water polygons.

**Vectors (~50 dimensions).** Densities per km² of land for bars, cafés,
restaurants, museums, theatres, shops by type, schools, healthcare, transit
stops, street segments by class, buildings, tall buildings, houses, apartment
blocks, offices, parks, sports facilities, mapped trees … plus land-use
fractions (residential, commercial, industrial, green, water, beach). See
`pipeline/features.py` for the exact tag mapping.

**Clustering.** Everything after extraction runs client-side (`js/ml.js`:
Jacobi PCA, k-means++). Clusters are numbered by average distance to the city
centre so borough 1 is always the most central; names are auto-generated from
each cluster's most distinctive features and can be edited in place.

## Run locally

```bash
python3 -m http.server 8000     # then open http://localhost:8000
```

No build step; the app is plain ES modules + MapLibre GL from a CDN.

## Add a city

```bash
python3 -m venv .venv && .venv/bin/pip install shapely requests numpy
# add an entry to pipeline/cities.json (OSM relation id + a "downtown" point)
.venv/bin/python pipeline/build_city.py <slug>
```

Extraction takes 2–6 minutes per city on the public Overpass API (responses are
cached under `pipeline/cache/`). `data/index.json` is regenerated automatically.

## Deploy

Pushing to `main` deploys via the GitHub Pages workflow in
`.github/workflows/pages.yml` (Settings → Pages → Source: *GitHub Actions*).

## Caveats

OSM completeness varies by city and by feature — trees, building heights and
land-use are mapped very unevenly. The clusters describe the *mapped* city.
Topos' original inputs also included image recognition, 311 complaints and
other proprietary signals that have no OSM analogue.

## Credits

Data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)
(ODbL). Basemap tiles by [CARTO](https://carto.com/attributions).
Concept by [Topos](https://medium.com/topos-ai).
