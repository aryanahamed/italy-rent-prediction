# -- CONSTANTS --
# This file contains constants used in the project.

from pathlib import Path

# PROJECT_ROOT — all data paths are relative to this.
# NOTE: Despite the name DATA_DIR, this is the PROJECT ROOT (Path(__file__).parent),
# not a 'data/' subdirectory. Subdirectories like 'data/', 'geocoding_cache/',
# and 'rent_prediction_model/' sit under this root.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  # alias for backward compatibility

# Subdirectory for raw/processed data files
DATA_SUBDIR = 'data'  # relative to PROJECT_ROOT

# Model settings
MARGIN_OF_ERROR = 0.1450747496597613

# Data contract exposed by the deployed model. Values outside these bounds are
# not represented by the source data used by the application and must not be
# silently extrapolated.
MIN_ROOMS = 1
MAX_ROOMS = 5
MIN_BATHROOMS = 1
MAX_BATHROOMS = 3
MIN_AREA_M2 = 30
MAX_AREA_M2 = 300

# Energy class configuration. Unknown is a real model category and should not
# be silently replaced with the previous default (D).
ENERGY_CLASSES = ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

ENERGY_CLASS_MAP = {
    'Unknown': 0.0,
    'A': 1.0,
    'B': 0.6,
    'C': 0.3,
    'D': 0.0,
    'E': -0.2,
    'F': -0.5,
    'G': -1.0
}

# Geocoding API configuration
PHOTON_API_BASE_URL = "https://photon.komoot.io/api/"
GEOCODING_TIMEOUT = 5  # seconds
GEOCODING_LIMIT = 5
PHOTON_COUNTRY_CODE = "IT"

# Current source/provenance limits. These features remain disabled until their
# underlying data contracts are rebuilt and validated.
DATA_AS_OF = "2023-12-07"
HISTORICAL_TRENDS_ENABLED = False
GEOGRAPHIC_MAPS_ENABLED = False

# Prediction analyzer settings
TOP_CONTRIBUTORS_COUNT = 5
PERTURBATION_SAMPLES = 30

# Heatmap visualization settings
HEATMAP_RADIUS = 25
HEATMAP_BLUR = 35
HEATMAP_MIN_OPACITY = 0.3
HEATMAP_MAX_ZOOM = 13

# Property cluster map settings
MAX_CLUSTER_PROPERTIES = 2000

# Italy coordinate bounds (used by map_data.py for coordinate validation)
ITALY_LAT_MIN = 35
ITALY_LAT_MAX = 47
ITALY_LON_MIN = 6
ITALY_LON_MAX = 19

# Convenience tuple for bounds checks: (lat_min, lat_max, lon_min, lon_max)
ITALY_BOUNDS = (ITALY_LAT_MIN, ITALY_LAT_MAX, ITALY_LON_MIN, ITALY_LON_MAX)

# Center of Italy (approximately Rome area) — used as default map center
ITALY_CENTER_COORDS = (41.9028, 12.4964)

# Minimum realistic rent for outlier filtering
MIN_RENT = 50.0
