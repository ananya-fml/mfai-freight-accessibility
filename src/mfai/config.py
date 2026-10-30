"""Study constants and locked analytical parameters.

Every value here is reported in Section 3 of the manuscript. Change one and
the pipeline no longer reproduces the paper.
"""

# ── Cities ──────────────────────────────────────────────────────────────
CITIES = ["patna", "varanasi", "indore", "kochi", "kolkata"]
CITY_ORDER = ["patna", "indore", "varanasi", "kochi", "kolkata"]  # figure order
CITY_LABEL = {c: c.title() for c in CITIES}

UTM = {"patna": 32645, "kolkata": 32645, "varanasi": 32644,
       "indore": 32643, "kochi": 32643}

CENTRES = {"patna": (25.6127, 85.1416), "indore": (22.7177, 75.8545),
           "varanasi": (25.3109, 83.0104), "kochi": (9.9816, 76.2830),
           "kolkata": (22.5697, 88.3497)}          # (lat, lon)

# Generous district bounds used to drop out-of-area geocodes (lon, lon, lat, lat)
DBOX = {
    "kochi":    (76.0, 77.2,  9.6, 10.4),
    "kolkata":  (88.0, 88.6, 22.3, 22.8),
    "patna":    (84.5, 86.2, 25.0, 25.9),
    "varanasi": (82.5, 83.3, 25.0, 25.7),
    "indore":   (75.3, 76.3, 22.2, 23.2),
}

# Indore has no inland-waterway terminal in-district; the nearest operational
# terminal is used instead (metres, name).
WATER_OVERRIDE = {
    "indore": (952_000, "Ramnagar Multi-Modal Terminal (Varanasi Port), NW-1"),
}

# ── Industry classification ─────────────────────────────────────────────
# ISIC Rev.4 Section C. Forced rather than chosen: the Kochi and Indore
# extracts arrived pre-filtered to manufacturing.
RESTRICT_DIVISIONS = list(range(10, 34))

MFG_GROUPS = {
    "G1_food_bev_tobacco":        [10, 11, 12],
    "G2_textile_apparel_leather": [13, 14, 15],
    "G3_wood_paper_printing":     [16, 17, 18],
    "G4_petro_chem_pharma":       [19, 20, 21],
    "G5_rubber_plastic_mineral":  [22, 23],
    "G6_metals":                  [24, 25],
    "G7_electronics_machinery":   [26, 27, 28],
    "G8_transport_furniture_oth": [29, 30, 31, 32, 33],
}
DIV_TO_GROUP = {d: g for g, ds in MFG_GROUPS.items() for d in ds}

GROUP_LABELS = {  # short code -> display label
    "G1": "Food, bev., tobacco",        "G2": "Textiles, apparel, leather",
    "G3": "Wood, paper, printing",      "G4": "Petro/chem/pharma",
    "G5": "Rubber/plastics/minerals",   "G6": "Basic/fab. metals",
    "G7": "Electronics/machinery",      "G8": "Transport/furniture/other",
}

# ── Network construction ────────────────────────────────────────────────
SNAP_TOL = 12.0          # m, endpoint snapping
ROUND_DP = 2
MAJOR_ROAD_TOL = 12.0    # m, node-to-major-road tagging
MAX_VERTEX_M = 25.0      # m, graph densification step
MAJOR_SAMPLE_M = 5.0     # m, major-road sampling step

# ── Geocoder-artefact filter ────────────────────────────────────────────
FALLBACK_THRESHOLD = 20        # min enterprises at one coordinate
FALLBACK_MIN_PINCODES = 5      # rule 1: city-centroid fallback
DIV_DIVERSITY_T = 8            # rule 2: pincode-centroid fallback

# ── MFAI ────────────────────────────────────────────────────────────────
# Externally anchored reference distances (m): access = T / (T + d)
T_REF = {"road": 2_000, "rail": 50_000, "water": 50_000, "icd": 50_000}

W_EQUAL = {"road": 0.25, "rail": 0.25, "icd": 0.25, "water": 0.25}
W_MODAL_SHARE = {"road": 0.674, "rail": 0.246, "icd": 0.045, "water": 0.035}
W_INFRA_ROLE = {"road": 0.35, "rail": 0.25, "icd": 0.35, "water": 0.05}
WEIGHTS_PRIMARY = W_EQUAL
WEIGHTS_ALT = {"modal_share": W_MODAL_SHARE, "infra_role": W_INFRA_ROLE}

COMPS = ["nearest_major_road_dist_m", "nearest_railway_dist_m",
         "nearest_waterway_dist_m", "nearest_ICD_dist_m"]
SHORT = ["road", "rail", "water", "icd"]
LAND_MODES = ["road", "rail", "icd"]   # MFAI-land: comparable across all cities

# ── Targetability ───────────────────────────────────────────────────────
POOLED_P10 = 0.7065      # pooled tenth percentile of MFAI-land
MCS, MS = 20, 10         # HDBSCAN min_cluster_size, min_samples
N_PERM = 999

# ── Commodity extension ─────────────────────────────────────────────────
LQ_CELL = {c: (1000 if c == "kolkata" else 2000) for c in CITIES}
LQ_MIN_FIRMS = 20
LQ_EB_K = 20
LQ_THR = 2.0
