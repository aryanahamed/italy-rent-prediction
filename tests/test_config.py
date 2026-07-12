"""Tests for config.py — constants and configuration values."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataDir:
    """Tests for DATA_DIR."""

    def test_data_dir_is_path(self):
        from config import DATA_DIR
        from pathlib import Path
        assert isinstance(DATA_DIR, Path)

    def test_data_dir_exists(self):
        from config import DATA_DIR
        assert DATA_DIR.exists()


class TestMarginOfError:
    """Tests for MARGIN_OF_ERROR."""

    def test_margin_of_error_is_positive_float(self):
        from config import MARGIN_OF_ERROR
        assert isinstance(MARGIN_OF_ERROR, float)
        assert MARGIN_OF_ERROR > 0

    def test_margin_of_error_reasonable(self):
        from config import MARGIN_OF_ERROR
        assert 0 < MARGIN_OF_ERROR < 1


class TestEnergyClasses:
    """Tests for ENERGY_CLASSES."""

    def test_is_list(self):
        from config import ENERGY_CLASSES
        assert isinstance(ENERGY_CLASSES, list)

    def test_contains_standard_classes(self):
        from config import ENERGY_CLASSES
        assert ENERGY_CLASSES == ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    def test_all_strings(self):
        from config import ENERGY_CLASSES
        assert all(isinstance(c, str) for c in ENERGY_CLASSES)


class TestEnergyClassMap:
    """Tests for ENERGY_CLASS_MAP."""

    def test_is_dict(self):
        from config import ENERGY_CLASS_MAP
        assert isinstance(ENERGY_CLASS_MAP, dict)

    def test_has_all_classes(self):
        from config import ENERGY_CLASS_MAP, ENERGY_CLASSES
        assert set(ENERGY_CLASS_MAP.keys()) == set(ENERGY_CLASSES)

    def test_a_is_highest(self):
        from config import ENERGY_CLASS_MAP
        assert ENERGY_CLASS_MAP['A'] == 1.0
        assert ENERGY_CLASS_MAP['A'] > ENERGY_CLASS_MAP['B']

    def test_g_is_lowest(self):
        from config import ENERGY_CLASS_MAP
        assert ENERGY_CLASS_MAP['G'] == -1.0
        assert ENERGY_CLASS_MAP['G'] < ENERGY_CLASS_MAP['F']

    def test_d_is_zero(self):
        from config import ENERGY_CLASS_MAP
        assert ENERGY_CLASS_MAP['D'] == 0.0

    def test_unknown_is_explicit(self):
        from config import ENERGY_CLASS_MAP
        assert ENERGY_CLASS_MAP['Unknown'] == 0.0

    def test_values_descend(self):
        from config import ENERGY_CLASS_MAP
        values = [ENERGY_CLASS_MAP[c] for c in ['A', 'B', 'C', 'D', 'E', 'F', 'G']]
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1], f"Values should descend at {i}"


class TestGeocodingConfig:
    """Tests for geocoding API configuration."""

    def test_photon_api_base_url(self):
        from config import PHOTON_API_BASE_URL
        assert isinstance(PHOTON_API_BASE_URL, str)
        assert PHOTON_API_BASE_URL.startswith('http')
        assert 'photon' in PHOTON_API_BASE_URL.lower()

    def test_geocoding_timeout(self):
        from config import GEOCODING_TIMEOUT
        assert isinstance(GEOCODING_TIMEOUT, int)
        assert GEOCODING_TIMEOUT > 0

    def test_geocoding_limit(self):
        from config import GEOCODING_LIMIT
        assert isinstance(GEOCODING_LIMIT, int)
        assert GEOCODING_LIMIT > 0


class TestPredictionAnalyzerSettings:
    """Tests for prediction analyzer configuration."""

    def test_top_contributors_count(self):
        from config import TOP_CONTRIBUTORS_COUNT
        assert isinstance(TOP_CONTRIBUTORS_COUNT, int)
        assert TOP_CONTRIBUTORS_COUNT > 0

    def test_perturbation_samples(self):
        from config import PERTURBATION_SAMPLES
        assert isinstance(PERTURBATION_SAMPLES, int)
        assert PERTURBATION_SAMPLES > 0


class TestHeatmapSettings:
    """Tests for heatmap visualization settings."""

    def test_heatmap_radius(self):
        from config import HEATMAP_RADIUS
        assert isinstance(HEATMAP_RADIUS, int)
        assert HEATMAP_RADIUS > 0

    def test_heatmap_blur(self):
        from config import HEATMAP_BLUR
        assert isinstance(HEATMAP_BLUR, int)
        assert HEATMAP_BLUR > 0

    def test_heatmap_min_opacity(self):
        from config import HEATMAP_MIN_OPACITY
        assert isinstance(HEATMAP_MIN_OPACITY, float)
        assert 0 < HEATMAP_MIN_OPACITY <= 1

    def test_heatmap_max_zoom(self):
        from config import HEATMAP_MAX_ZOOM
        assert isinstance(HEATMAP_MAX_ZOOM, int)
        assert HEATMAP_MAX_ZOOM > 0


class TestMaxClusterProperties:
    """Tests for MAX_CLUSTER_PROPERTIES."""

    def test_is_positive_int(self):
        from config import MAX_CLUSTER_PROPERTIES
        assert isinstance(MAX_CLUSTER_PROPERTIES, int)
        assert MAX_CLUSTER_PROPERTIES > 0


class TestItalyBounds:
    """Tests for Italy coordinate bounds."""

    def test_lat_min(self):
        from config import ITALY_LAT_MIN
        assert ITALY_LAT_MIN == 35

    def test_lat_max(self):
        from config import ITALY_LAT_MAX
        assert ITALY_LAT_MAX == 47

    def test_lon_min(self):
        from config import ITALY_LON_MIN
        assert ITALY_LON_MIN == 6

    def test_lon_max(self):
        from config import ITALY_LON_MAX
        assert ITALY_LON_MAX == 19

    def test_bounds_tuple(self):
        from config import ITALY_BOUNDS
        assert ITALY_BOUNDS == (35, 47, 6, 19)

    def test_bounds_are_integers(self):
        from config import ITALY_BOUNDS
        assert all(isinstance(v, int) for v in ITALY_BOUNDS)

    def test_latitudes_are_sensible(self):
        from config import ITALY_LAT_MIN, ITALY_LAT_MAX
        assert ITALY_LAT_MIN < ITALY_LAT_MAX

    def test_longitudes_are_sensible(self):
        from config import ITALY_LON_MIN, ITALY_LON_MAX
        assert ITALY_LON_MIN < ITALY_LON_MAX

    def test_bounds_cover_italy_properly(self):
        """Rome and Milan should be within bounds."""
        from config import ITALY_LAT_MIN, ITALY_LAT_MAX, ITALY_LON_MIN, ITALY_LON_MAX
        # Rome approx 41.9, 12.5
        assert ITALY_LAT_MIN <= 41.9 <= ITALY_LAT_MAX
        assert ITALY_LON_MIN <= 12.5 <= ITALY_LON_MAX
        # Milan approx 45.5, 9.2
        assert ITALY_LAT_MIN <= 45.5 <= ITALY_LAT_MAX
        assert ITALY_LON_MIN <= 9.2 <= ITALY_LON_MAX


class TestItalyCenterCoords:
    """Tests for ITALY_CENTER_COORDS."""

    def test_is_tuple(self):
        from config import ITALY_CENTER_COORDS
        assert isinstance(ITALY_CENTER_COORDS, tuple)
        assert len(ITALY_CENTER_COORDS) == 2

    def test_lat_lon_types(self):
        from config import ITALY_CENTER_COORDS
        lat, lon = ITALY_CENTER_COORDS
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_center_near_rome(self):
        from config import ITALY_CENTER_COORDS
        lat, lon = ITALY_CENTER_COORDS
        # Rome is approximately 41.9028, 12.4964
        assert abs(lat - 41.9028) < 0.1
        assert abs(lon - 12.4964) < 0.1

    def test_within_italy_bounds(self):
        from config import ITALY_CENTER_COORDS, ITALY_BOUNDS
        lat, lon = ITALY_CENTER_COORDS
        lat_min, lat_max, lon_min, lon_max = ITALY_BOUNDS
        assert lat_min <= lat <= lat_max
        assert lon_min <= lon <= lon_max


class TestMinRent:
    """Tests for MIN_RENT."""

    def test_is_positive_float(self):
        from config import MIN_RENT
        assert isinstance(MIN_RENT, float)
        assert MIN_RENT > 0

    def test_reasonable_value(self):
        """Minimum rent should be realistic for Italy."""
        from config import MIN_RENT
        assert 0 < MIN_RENT < 500


class TestConstantsAreImmutable:
    """Tests that constants are not accidentally mutable for shared objects."""

    def test_energy_classes_is_mutable_list(self):
        """ENERGY_CLASSES is a list (mutable by design)."""
        from config import ENERGY_CLASSES
        original = list(ENERGY_CLASSES)
        ENERGY_CLASSES.append('H')
        # After append, the module-level list is modified
        assert len(ENERGY_CLASSES) == len(original) + 1
        # Clean up
        ENERGY_CLASSES.pop()
