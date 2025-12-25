# -- CONSTANTS --
# This file contains constants used in the project.

# Model settings
MARGIN_OF_ERROR = 0.1450747496597613

# Energy class configuration
ENERGY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

ENERGY_CLASS_MAP = {
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

# Italy coordinate bounds
ITALY_LAT_MIN = 35
ITALY_LAT_MAX = 47
ITALY_LON_MIN = 6
ITALY_LON_MAX = 19