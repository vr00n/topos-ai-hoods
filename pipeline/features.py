"""
Feature definitions for the neighborhood vectors.

Every feature is derived from OpenStreetMap tags. Point-like features are
counted per hex cell and converted to densities (count per km^2 of land).
Area features are fractions of the cell's land area covered by a land-use
class.

Each Overpass "source" is a single query that returns a compact CSV of
(lat, lon, tag columns...). Rows are bucketed into features locally.
"""

# --- Point sources -----------------------------------------------------------
# name -> dict(query=<overpass statements using (area.a) and {bbox}>, cols=[...])
POINT_SOURCES = {
    "amenity": {
        "cols": ["amenity"],
        "query": (
            'nwr["amenity"~"^(bar|pub|nightclub|biergarten|restaurant|cafe|fast_food|ice_cream|'
            'theatre|cinema|arts_centre|library|place_of_worship|school|kindergarten|childcare|'
            'university|college|hospital|doctors|clinic|dentist|pharmacy|bank|atm|post_office|'
            'police|fire_station|parking|fuel|charging_station|bicycle_parking|bicycle_rental|'
            'community_centre|social_facility|marketplace|veterinary|car_wash|casino|stripclub|'
            'coworking_space|events_venue|music_venue)$"](area.a){bbox};'
        ),
    },
    "shop": {
        "cols": ["shop"],
        "query": 'nwr["shop"](area.a){bbox};',
    },
    "tlo": {  # tourism / leisure / office
        "cols": ["tourism", "leisure", "office"],
        "query": (
            'nwr["tourism"](area.a){bbox};'
            'nwr["leisure"](area.a){bbox};'
            'nwr["office"](area.a){bbox};'
        ),
    },
    "transit": {
        "cols": ["railway", "highway", "station"],
        "query": (
            'nwr["railway"~"^(station|halt|tram_stop|subway_entrance)$"](area.a){bbox};'
            'node["highway"="bus_stop"](area.a){bbox};'
        ),
    },
    "trees": {
        "cols": [],
        "query": 'node["natural"="tree"](area.a){bbox};',
    },
    "streets": {
        "cols": ["highway"],
        "query": (
            'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|'
            'living_street|pedestrian|footway|steps|cycleway|path|service)$"](area.a){bbox};'
        ),
    },
    "buildings": {
        "cols": ["building", "building:levels", "height"],
        "query": 'way["building"](area.a){bbox};',
    },
}

# --- Land-use polygon source ---------------------------------------------------
LANDUSE_QUERY = (
    'nwr["landuse"~"^(residential|commercial|retail|industrial|grass|forest|meadow|cemetery|'
    'farmland|recreation_ground|orchard|vineyard|allotments|village_green|railway|military|'
    'brownfield|construction)$"](area.a){bbox};'
    'nwr["leisure"~"^(park|garden|golf_course|nature_reserve|pitch)$"](area.a){bbox};'
    'nwr["natural"~"^(wood|water|scrub|wetland|beach|heath|grassland|bay|sand)$"](area.a){bbox};'
    'nwr["aeroway"="aerodrome"](area.a){bbox};'
)

# landuse class -> list of (key, value-set or None for any)
LANDUSE_CLASSES = {
    "residential": [("landuse", {"residential"})],
    "commercial": [("landuse", {"commercial", "retail"})],
    "industrial": [("landuse", {"industrial", "railway", "military", "brownfield", "construction"})],
    "green": [
        ("landuse", {"grass", "forest", "meadow", "cemetery", "farmland", "recreation_ground",
                     "orchard", "vineyard", "allotments", "village_green"}),
        ("leisure", {"park", "garden", "golf_course", "nature_reserve", "pitch", "playground", "dog_park"}),
        ("natural", {"wood", "scrub", "wetland", "heath", "grassland"}),
    ],
    "water": [("natural", {"water", "bay"})],
    "beach": [("natural", {"beach", "sand"})],
    "airport": [("aeroway", {"aerodrome"})],
}


def _levels(v):
    try:
        return float(str(v).split(";")[0].replace(",", "."))
    except Exception:
        return None


def _height_m(v):
    if not v:
        return None
    s = str(v).strip().lower()
    try:
        if s.endswith("m"):
            s = s[:-1].strip()
        if "'" in s or "ft" in s:
            return float(s.split("'")[0].replace("ft", "").strip()) * 0.3048
        return float(s.split(";")[0].replace(",", "."))
    except Exception:
        return None


# --- Point features -------------------------------------------------------------
# key, label, group, source, matcher(row: dict) -> bool
def _in(col, values):
    vs = set(values)
    return lambda r: r.get(col) in vs


def _any(col):
    return lambda r: bool(r.get(col))


def _tall(r):
    lv = _levels(r.get("building:levels"))
    if lv is not None and lv >= 6:
        return True
    h = _height_m(r.get("height"))
    return h is not None and h >= 20


POINT_FEATURES = [
    # Nightlife
    ("bars_pubs", "Bars & pubs", "Nightlife", "amenity", _in("amenity", ["bar", "pub", "biergarten"])),
    ("nightclubs", "Nightclubs & music venues", "Nightlife", "amenity",
     _in("amenity", ["nightclub", "casino", "stripclub", "music_venue", "events_venue"])),
    ("liquor_stores", "Liquor & wine stores", "Nightlife", "shop", _in("shop", ["alcohol", "wine"])),
    # Food & drink
    ("restaurants", "Restaurants", "Food & Drink", "amenity", _in("amenity", ["restaurant"])),
    ("cafes", "Cafes", "Food & Drink", "amenity", _in("amenity", ["cafe"])),
    ("fast_food", "Fast food & ice cream", "Food & Drink", "amenity", _in("amenity", ["fast_food", "ice_cream"])),
    ("bakeries_delis", "Bakeries & delis", "Food & Drink", "shop",
     _in("shop", ["bakery", "deli", "butcher", "cheese", "pastry", "confectionery", "coffee", "tea"])),
    # Culture
    ("museums_galleries", "Museums & galleries", "Culture", "tlo", _in("tourism", ["museum", "gallery"])),
    ("theatres_cinemas", "Theatres & cinemas", "Culture", "amenity", _in("amenity", ["theatre", "cinema", "arts_centre"])),
    ("libraries", "Libraries", "Culture", "amenity", _in("amenity", ["library"])),
    ("bookshops", "Bookshops", "Culture", "shop", _in("shop", ["books"])),
    ("places_of_worship", "Places of worship", "Culture", "amenity", _in("amenity", ["place_of_worship"])),
    ("public_art", "Public art", "Culture", "tlo", _in("tourism", ["artwork"])),
    ("attractions", "Tourist attractions", "Culture", "tlo", _in("tourism", ["attraction", "viewpoint"])),
    ("hotels", "Hotels & hostels", "Culture", "tlo", _in("tourism", ["hotel", "hostel", "guest_house", "motel", "apartment"])),
    # Retail
    ("shops_total", "All shops", "Retail", "shop", _any("shop")),
    ("grocery", "Grocery & convenience", "Retail", "shop",
     _in("shop", ["supermarket", "convenience", "greengrocer", "general", "grocery"])),
    ("fashion", "Fashion & boutiques", "Retail", "shop",
     _in("shop", ["clothes", "shoes", "jewelry", "boutique", "fashion_accessories", "bag", "watches"])),
    ("personal_care", "Salons & personal care", "Retail", "shop",
     _in("shop", ["hairdresser", "beauty", "massage", "tattoo", "cosmetics", "nail_salon"])),
    ("laundry_repair", "Laundry & repair shops", "Retail", "shop",
     _in("shop", ["laundry", "dry_cleaning", "hardware", "doityourself", "car_repair", "mobile_phone", "variety_store"])),
    # Civic
    ("schools", "Schools & childcare", "Civic", "amenity", _in("amenity", ["school", "kindergarten", "childcare"])),
    ("higher_ed", "Colleges & universities", "Civic", "amenity", _in("amenity", ["university", "college"])),
    ("healthcare", "Healthcare", "Civic", "amenity", _in("amenity", ["hospital", "doctors", "clinic", "dentist", "pharmacy"])),
    ("banks_atms", "Banks & ATMs", "Civic", "amenity", _in("amenity", ["bank", "atm"])),
    ("public_safety", "Police & fire", "Civic", "amenity", _in("amenity", ["police", "fire_station", "post_office"])),
    ("community", "Community & social services", "Civic", "amenity",
     _in("amenity", ["community_centre", "social_facility", "marketplace", "veterinary"])),
    # Transit & mobility
    ("rail_stations", "Rail & subway stations", "Transit", "transit", _in("railway", ["station", "halt"])),
    ("subway_entrances", "Subway entrances", "Transit", "transit", _in("railway", ["subway_entrance"])),
    ("tram_stops", "Tram stops", "Transit", "transit", _in("railway", ["tram_stop"])),
    ("bus_stops", "Bus stops", "Transit", "transit", _in("highway", ["bus_stop"])),
    ("bike_infra", "Bike parking & rental", "Transit", "amenity", _in("amenity", ["bicycle_parking", "bicycle_rental"])),
    ("car_parking", "Car parking", "Transit", "amenity", _in("amenity", ["parking"])),
    ("fuel_charging", "Fuel & EV charging", "Transit", "amenity", _in("amenity", ["fuel", "charging_station", "car_wash"])),
    # Streets
    ("major_roads", "Major road segments", "Streets", "streets",
     _in("highway", ["motorway", "trunk", "primary", "secondary", "tertiary"])),
    ("local_streets", "Local street segments", "Streets", "streets",
     _in("highway", ["residential", "unclassified", "living_street"])),
    ("pedestrian_paths", "Pedestrian paths", "Streets", "streets", _in("highway", ["pedestrian", "footway", "steps", "path"])),
    ("cycleways", "Cycleways", "Streets", "streets", _in("highway", ["cycleway"])),
    ("service_roads", "Service roads & driveways", "Streets", "streets", _in("highway", ["service"])),
    # Built form
    ("buildings", "Buildings", "Built Form", "buildings", _any("building")),
    ("tall_buildings", "Tall buildings (6+ floors / 20m+)", "Built Form", "buildings", _tall),
    ("houses", "Single-family houses", "Built Form", "buildings",
     _in("building", ["house", "detached", "semidetached_house", "terrace", "bungalow", "semi"])),
    ("apartments", "Apartment buildings", "Built Form", "buildings", _in("building", ["apartments", "residential", "dormitory"])),
    ("offices", "Offices", "Built Form", "tlo", _any("office")),
    ("coworking", "Coworking spaces", "Built Form", "amenity", _in("amenity", ["coworking_space"])),
    # Nature & leisure
    ("parks", "Parks", "Nature", "tlo", _in("leisure", ["park", "nature_reserve"])),
    ("gardens_playgrounds", "Gardens & playgrounds", "Nature", "tlo", _in("leisure", ["garden", "playground", "dog_park"])),
    ("sports", "Sports facilities", "Nature", "tlo",
     _in("leisure", ["pitch", "sports_centre", "fitness_centre", "swimming_pool", "golf_course", "track", "stadium"])),
    ("trees", "Mapped trees", "Nature", "trees", lambda r: True),
]

# --- Area features (fractions) ----------------------------------------------------
AREA_FEATURES = [
    ("residential_frac", "Residential land-use", "Built Form", "residential"),
    ("commercial_frac", "Commercial / retail land-use", "Built Form", "commercial"),
    ("industrial_frac", "Industrial / rail land-use", "Built Form", "industrial"),
    ("green_frac", "Green space", "Nature", "green"),
    ("water_frac", "Inland water", "Nature", "water"),
    ("beach_frac", "Beach & sand", "Nature", "beach"),
]

GROUPS = ["Nightlife", "Food & Drink", "Culture", "Retail", "Civic", "Transit", "Streets", "Built Form", "Nature"]
