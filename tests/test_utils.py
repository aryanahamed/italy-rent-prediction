"""Tests for utils.py — remove_price_outliers."""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import remove_price_outliers


# =============================================================================
# remove_price_outliers tests — covers MU-2, MU-3, MU-8
# =============================================================================

class TestRemovePriceOutliers:
    """Tests for remove_price_outliers."""

    def test_empty_dataframe(self):
        df = pd.DataFrame({'price': []})
        result = remove_price_outliers(df)
        assert len(result) == 0

    def test_missing_column_raises_contract_error(self):
        df = pd.DataFrame({'other': [1, 2, 3]})
        with pytest.raises(ValueError, match="Required outlier column 'price'"):
            remove_price_outliers(df, column='price')

    def test_all_nan_column(self):
        df = pd.DataFrame({'price': [np.nan, np.nan]})
        result = remove_price_outliers(df)
        assert len(result) == 0  # After NaN removal, empty DataFrame returned

    def test_no_outliers(self):
        """Normal data with no outliers returns all rows (IQR)."""
        df = pd.DataFrame({'price': [500, 600, 700, 800, 900, 1000]})
        result = remove_price_outliers(df, method='iqr')
        assert len(result) == 6

    def test_removes_high_outliers(self):
        """Extreme high values removed by IQR."""
        df = pd.DataFrame({'price': [500, 600, 700, 800, 900, 10000]})
        result = remove_price_outliers(df, method='iqr')
        assert len(result) == 5
        assert 10000 not in result['price'].values

    def test_removes_low_outliers(self):
        """Extreme low values removed by IQR."""
        df = pd.DataFrame({'price': [1, 600, 700, 800, 900, 1000]})
        result = remove_price_outliers(df, method='iqr')
        assert len(result) == 5
        assert 1 not in result['price'].values

    def test_min_rent_filter_applied(self):
        """Prices below min_rent filtered out (MU-8)."""
        df = pd.DataFrame({'price': [1, 5, 10, 600, 700, 800]})
        result = remove_price_outliers(df, method='iqr', min_rent=50)
        assert 1 not in result['price'].values
        assert 5 not in result['price'].values
        assert 10 not in result['price'].values
        assert 600 in result['price'].values

    def test_zscore_method_removes_outliers(self):
        """zscore method removes 3-sigma outliers (MU-2)."""
        # Create data with one extreme outlier
        np.random.seed(42)
        prices = list(np.random.normal(800, 100, 100)) + [5000]
        df = pd.DataFrame({'price': prices})
        result = remove_price_outliers(df, method='zscore')
        assert len(result) < len(df)
        assert 5000 not in result['price'].values

    def test_zscore_returns_dataframe(self):
        df = pd.DataFrame({'price': [500, 600, 700, 800, 900]})
        result = remove_price_outliers(df, method='zscore')
        assert isinstance(result, pd.DataFrame)

    def test_unsupported_method_raises_value_error(self):
        """Unknown method raises ValueError (MU-2 fix)."""
        df = pd.DataFrame({'price': [500, 600, 700]})
        with pytest.raises(ValueError, match='Unsupported method'):
            remove_price_outliers(df, method='not_a_method')

    def test_iqr_and_zscore_are_consistent_types(self):
        """Both methods return consistent types (MU-3)."""
        df = pd.DataFrame({'price': [500, 600, 700, 800, 900, 1000]})
        iqr_result = remove_price_outliers(df, method='iqr')
        zscore_result = remove_price_outliers(df, method='zscore')
        assert isinstance(iqr_result, pd.DataFrame)
        assert isinstance(zscore_result, pd.DataFrame)

    def test_single_value_column(self):
        """Single value doesn't crash (IQR=0 edge case)."""
        df = pd.DataFrame({'price': [800, 800, 800, 800]})
        result = remove_price_outliers(df, method='iqr')
        assert len(result) == 4  # All identical, no outliers

    def test_single_row(self):
        df = pd.DataFrame({'price': [800]})
        result = remove_price_outliers(df, method='iqr')
        assert len(result) == 1

    def test_zscore_with_zero_std(self):
        """zscore with no variance returns all rows unchanged."""
        df = pd.DataFrame({'price': [800, 800, 800]})
        result = remove_price_outliers(df, method='zscore')
        assert len(result) == 3

    def test_zscore_zero_std_still_applies_minimum_rent(self):
        df = pd.DataFrame({'price': [20, 20, 20]})
        result = remove_price_outliers(df, method='zscore', min_rent=50)
        assert result.empty

    def test_negative_prices_filtered_by_min_rent(self):
        """Negative prices filtered below min_rent (MU-8)."""
        df = pd.DataFrame({'price': [-100, -50, 0, 600, 700]})
        result = remove_price_outliers(df, method='iqr', min_rent=50)
        assert all(p >= 50 for p in result['price'].values)

    def test_default_min_rent_applied(self):
        """Default min_rent=50 filters extremely cheap prices."""
        df = pd.DataFrame({'price': [1, 2, 3, 500, 600, 700]})
        result = remove_price_outliers(df, method='iqr')
        assert all(p >= 50 for p in result['price'].values)

    def test_custom_min_rent(self):
        df = pd.DataFrame({'price': [100, 200, 500, 600, 700]})
        result = remove_price_outliers(df, method='iqr', min_rent=250)
        assert 100 not in result['price'].values
        assert 200 not in result['price'].values
