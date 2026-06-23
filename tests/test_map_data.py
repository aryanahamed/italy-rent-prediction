"""Tests for map_data.py — get_price_category (pure logic, no Streamlit dependency)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from map_data import get_price_category


# =============================================================================
# get_price_category tests
# =============================================================================

class TestGetPriceCategory:
    """Tests for get_price_category — boundary logic."""

    def test_budget(self):
        result = get_price_category(300)
        assert result['color'] == 'green'
        assert 'Budget' in result['label']

    def test_budget_upper_boundary(self):
        result = get_price_category(499)
        assert result['color'] == 'green'

    def test_affordable_lower_boundary(self):
        result = get_price_category(500)
        assert result['color'] == 'blue'
        assert 'Affordable' in result['label']

    def test_affordable_upper_boundary(self):
        result = get_price_category(999)
        assert result['color'] == 'blue'

    def test_mid_range_lower_boundary(self):
        result = get_price_category(1000)
        assert result['color'] == 'orange'
        assert 'Mid-range' in result['label']

    def test_mid_range_upper_boundary(self):
        result = get_price_category(1499)
        assert result['color'] == 'orange'

    def test_premium_lower_boundary(self):
        result = get_price_category(1500)
        assert result['color'] == 'red'
        assert 'Premium' in result['label']

    def test_premium_upper_boundary(self):
        result = get_price_category(2499)
        assert result['color'] == 'red'

    def test_luxury_lower_boundary(self):
        result = get_price_category(2500)
        assert result['color'] == 'purple'
        assert 'Luxury' in result['label']

    def test_luxury_high_value(self):
        result = get_price_category(10000)
        assert result['color'] == 'purple'

    def test_negative_price_falls_through_to_budget(self):
        """Negative prices (not in data, but handled gracefully)."""
        result = get_price_category(-100)
        assert result['color'] == 'green'
        assert 'Budget' in result['label']

    def test_zero_price(self):
        result = get_price_category(0)
        assert result['color'] == 'green'

    def test_all_return_includes_color_label_and_icon(self):
        keys = {'color', 'label', 'icon'}
        for price in [100, 500, 1000, 1500, 2500]:
            result = get_price_category(price)
            assert keys.issubset(result.keys())
