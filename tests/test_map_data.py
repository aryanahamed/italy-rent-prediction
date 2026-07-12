"""Tests for map_data.py — pure logic and mocked I/O."""

import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import streamlit as st
from map_data import _validate_coords, get_price_category
from utils import COLUMN_NAMES


# ── _validate_coords ──────────────────────────────────────────────────────────

class TestValidateCoords:
    def test_rome(self):           assert _validate_coords(41.9, 12.5)
    def test_milan(self):          assert _validate_coords(45.46, 9.19)
    def test_naples(self):         assert _validate_coords(40.85, 14.27)
    def test_london(self):         assert not _validate_coords(51.5, -0.1)
    def test_nyc(self):            assert not _validate_coords(40.7, -74.0)
    def test_lat_below(self):      assert not _validate_coords(34.9, 12.0)
    def test_lat_above(self):      assert not _validate_coords(47.1, 12.0)
    def test_lon_below(self):      assert not _validate_coords(41.9, 5.9)
    def test_lon_above(self):      assert not _validate_coords(41.9, 19.1)
    def test_min_edge(self):       assert _validate_coords(35.0, 6.0)
    def test_max_edge(self):       assert _validate_coords(47.0, 19.0)


# ── get_price_category ────────────────────────────────────────────────────────

class TestGetPriceCategory:
    def test_lowest_band(self): r = get_price_category(300); assert r['color'] == 'green' and r['label'] == 'Under €500'
    def test_affordable(self): r = get_price_category(750);  assert r['color'] == 'blue'
    def test_mid(self):        r = get_price_category(1250); assert r['color'] == 'orange'
    def test_premium(self):    r = get_price_category(2000); assert r['color'] == 'red'
    def test_luxury(self):     r = get_price_category(3000); assert r['color'] == 'purple'
    def test_boundaries(self):
        assert get_price_category(499)['color'] == 'green'
        assert get_price_category(500)['color'] == 'blue'
        assert get_price_category(999)['color'] == 'blue'
        assert get_price_category(1000)['color'] == 'orange'
        assert get_price_category(1499)['color'] == 'orange'
        assert get_price_category(1500)['color'] == 'red'
        assert get_price_category(2499)['color'] == 'red'
        assert get_price_category(2500)['color'] == 'purple'
    def test_negative(self):
        with pytest.raises(ValueError, match='non-negative'):
            get_price_category(-100)
    def test_zero(self):       assert get_price_category(0)['color'] == 'green'
    def test_keys(self):
        for p in [0, 300, 750, 1250, 2000, 5000]:
            r = get_price_category(p)
            assert 'color' in r and 'label' in r and 'icon' in r


# ── Helper: rental DataFrame with COLUMN_NAMES-order columns ──────────────────

def _rental_df(**kw):
    """Return n-row DataFrame with columns matching COLUMN_NAMES order."""
    import pandas as pd
    defaults = {
        'region': 'Lazio', 'city': 'Roma', 'neighborhood': 'Centro',
        'price': 1000, 'datetime': '2023-01-01', 'parking_spots': 0,
        'bathrooms_per_room': 0.5, 'bathrooms': 1, 'rooms': 2,
        'top_floor': 0, 'condition': 'good', 'energy_class': 'C',
        'sea_view': 0, 'central_heating': 1, 'area': 80.0,
        'furnished': 1, 'balcony': 1, 'tv_system': 0,
        'external_exposure': 1, 'fiber_optic': 1, 'electric_gate': 0,
        'cellar': 0, 'shared_garden': 0, 'private_garden': 0,
        'alarm_system': 0, 'concierge': 0, 'pool': 0,
        'villa': 0, 'entire_property': 0, 'apartment': 1,
        'penthouse': 0, 'loft': 0, 'attic': 0,
    }
    defaults.update(kw)
    n = max((len(v) for v in defaults.values() if isinstance(v, list)), default=1)
    data = {}
    for c in COLUMN_NAMES:
        v = defaults[c]
        data[c] = v if isinstance(v, list) else [v] * n
    return pd.DataFrame(data)


def _setup_mocks(monkeypatch, coords_json, df):
    """Mock open and pd.read_csv so cached loaders return our test data."""
    import builtins, json
    class MockFile:
        def __init__(self, c): self.content = c
        def read(self, *a): return self.content
        def __enter__(self): return self
        def __exit__(self, *a): pass
    orig = builtins.open
    def mock_open(path, *a, **kw):
        if 'neighborhood_coordinates.json' in str(path):
            return MockFile(coords_json)
        return orig(path, *a, **kw)
    monkeypatch.setattr(builtins, 'open', mock_open)
    import pandas as pd
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **kw: df)
    st.cache_data.clear()


def _setup_mocks_coords_fail(monkeypatch):
    """Mock open so load_neighborhood_coordinates raises FileNotFoundError."""
    import builtins
    orig = builtins.open
    def mock_open(path, *a, **kw):
        if 'neighborhood_coordinates.json' in str(path):
            raise FileNotFoundError('mock')
        return orig(path, *a, **kw)
    monkeypatch.setattr(builtins, 'open', mock_open)
    st.cache_data.clear()


def _setup_mocks_rental_fail(monkeypatch):
    """Mock so load_neighborhood_coordinates works but load_rental_data fails."""
    import builtins, json
    class M:
        def read(self, *a): return '{"A":[41.9,12.5]}'
        def __enter__(self): return self
        def __exit__(self, *a): pass
    orig = builtins.open
    def mock_open(path, *a, **kw):
        p = str(path)
        if 'neighborhood_coordinates.json' in p:
            return M()
        if 'rents_clean' in p and p.endswith('.csv'):
            raise FileNotFoundError('mock')
        return orig(path, *a, **kw)
    monkeypatch.setattr(builtins, 'open', mock_open)
    st.cache_data.clear()


# ── load_neighborhood_price_data tests ────────────────────────────────────────

class TestLoadNeighborhoodPriceData:
    def test_coords_not_found_returns_empty(self, monkeypatch):
        _setup_mocks_coords_fail(monkeypatch)
        from map_data import load_neighborhood_price_data
        assert load_neighborhood_price_data() == ([], {}, [])

    def test_rental_not_found_returns_empty(self, monkeypatch):
        _setup_mocks_rental_fail(monkeypatch)
        from map_data import load_neighborhood_price_data
        assert load_neighborhood_price_data() == ([], {}, [])

    def test_valid_data_returns_heatmap_stats_neighborhoods(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[1000, 2000], city=['Roma', 'Milano']))
        from map_data import load_neighborhood_price_data
        hd, stats, valid = load_neighborhood_price_data()
        assert len(hd) == 2
        assert stats['neighborhood_count'] == 2
        assert stats['avg_price'] == pytest.approx(1500.0)
        assert stats['min_price'] == pytest.approx(1000.0)
        assert stats['max_price'] == pytest.approx(2000.0)
        assert len(valid) == 2

    def test_stats_in_euros_not_normalized(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[800, 1200]))
        from map_data import load_neighborhood_price_data
        _, stats, _ = load_neighborhood_price_data()
        assert stats['avg_price'] == pytest.approx(1000.0)
        assert stats['min_price'] == pytest.approx(800.0)
        assert stats['max_price'] == pytest.approx(1200.0)

    def test_neighborhood_missing_from_coords_skipped(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"Centro":[41.9,12.5]}',
            _rental_df(neighborhood=['Centro', 'Unknown'], price=[1000, 800], city=['Roma', 'Milano']))
        from map_data import load_neighborhood_price_data
        hd, _, valid = load_neighborhood_price_data()
        assert len(hd) == 1
        assert valid[0]['neighborhood'] == 'Centro'

    def test_same_price_normalizes_to_zero(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[1000, 1000]))
        from map_data import load_neighborhood_price_data
        hd, _, _ = load_neighborhood_price_data()
        assert all(p[2] == 0.0 for p in hd)

    def test_high_valid_rent_is_not_removed_as_global_outlier(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[1000, 20000]))
        from map_data import load_neighborhood_price_data
        _, stats, _ = load_neighborhood_price_data()
        assert stats['max_price'] == 20000


# ── load_property_cluster_data tests ──────────────────────────────────────────

class TestLoadPropertyClusterData:
    def test_coords_not_found_returns_empty(self, monkeypatch):
        _setup_mocks_coords_fail(monkeypatch)
        from map_data import load_property_cluster_data
        assert load_property_cluster_data() == []

    def test_rental_not_found_returns_empty(self, monkeypatch):
        _setup_mocks_rental_fail(monkeypatch)
        from map_data import load_property_cluster_data
        assert load_property_cluster_data() == []

    def test_valid_data(self, monkeypatch):
        _setup_mocks(monkeypatch, '{"Centro":[41.9,12.5]}',
                     _rental_df(neighborhood=['Centro'], city=['Roma']))
        from map_data import load_property_cluster_data
        props = load_property_cluster_data()
        assert len(props) == 1
        assert props[0]['city'] == 'Roma'
        assert props[0]['neighborhood'] == 'Centro'

    def test_price_filter(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[500, 2000],
                       city=['Roma', 'Milano'], rooms=[2, 3]))
        from map_data import load_property_cluster_data
        assert len(load_property_cluster_data(price_min=600, price_max=1500)) == 0

    def test_rooms_filter(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[1000, 1200],
                       city=['Roma', 'Milano'], rooms=[1, 5]))
        from map_data import load_property_cluster_data
        assert len(load_property_cluster_data(rooms_min=2, rooms_max=4)) == 0

    def test_both_filters(self, monkeypatch):
        _setup_mocks(monkeypatch,
            '{"A":[41.9,12.5],"B":[42.0,12.6]}',
            _rental_df(neighborhood=['A', 'B'], price=[500, 1200],
                       city=['Roma', 'Milano'], rooms=[1, 3]))
        from map_data import load_property_cluster_data
        props = load_property_cluster_data(price_min=400, price_max=1500, rooms_min=2, rooms_max=4)
        assert len(props) == 1
        assert props[0]['price'] == 1200

    def test_unknown_neighborhood_skipped(self, monkeypatch):
        _setup_mocks(monkeypatch, '{"Centro":[41.9,12.5]}',
                     _rental_df(neighborhood=['Unknown'], city=['Roma']))
        from map_data import load_property_cluster_data
        assert load_property_cluster_data() == []
