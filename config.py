# -- CONSTANTS --
# This file contains constants used in the project.
MARGIN_OF_ERROR = 0.1450747496597613

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

PHOTON_API_URL = "https://photon.komoot.io/api/?q={query} Italy&limit=5"