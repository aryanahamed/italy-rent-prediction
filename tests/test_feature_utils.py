"""Tests for feature_utils.py — affordability, report generation, and similar properties."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_utils import (
    calculate_affordability,
    generate_prediction_report,
)


# =============================================================================
# calculate_affordability tests
# =============================================================================

class TestCalculateAffordability:
    """Tests for calculate_affordability."""

    def test_returns_dict_with_expected_keys(self):
        result = calculate_affordability(800)
        expected_keys = [
            'rent_price', 'required_income', 'avg_salary',
            'pct_of_avg_salary', 'affordability_level', 'affordability_color',
            'affordability_emoji', 'is_affordable'
        ]
        for key in expected_keys:
            assert key in result

    def test_highly_affordable(self):
        # 30% of 1650 = 495 → rent <= 495 is highly affordable
        result = calculate_affordability(400)
        assert result['affordability_level'] == 'Highly Affordable'
        assert result['is_affordable'] is True

    def test_affordable_boundary(self):
        # 30% of 1650 = 495
        result = calculate_affordability(495)
        assert result['affordability_level'] == 'Highly Affordable'

    def test_affordable(self):
        # Between 30% and 40%
        result = calculate_affordability(600)
        assert result['affordability_level'] == 'Affordable'

    def test_moderate(self):
        # Between 40% and 50%
        result = calculate_affordability(750)
        assert result['affordability_level'] == 'Moderate'

    def test_challenging(self):
        # Over 50%
        result = calculate_affordability(1000)
        assert result['affordability_level'] == 'Challenging'
        assert result['is_affordable'] is False

    def test_requires_correct_income(self):
        result = calculate_affordability(600)
        assert result['required_income'] == pytest.approx(2000.0)  # 600 / 0.30

    def test_uses_national_salary_by_default(self):
        result = calculate_affordability(800)
        assert result['avg_salary'] == 1650.0
        assert result['pct_of_avg_salary'] == pytest.approx(800 / 1650 * 100)

    def test_uses_regional_salary(self):
        result = calculate_affordability(1000, region='Lombardia')
        assert result['avg_salary'] == 2000.0
        assert result['pct_of_avg_salary'] == pytest.approx(1000 / 2000 * 100)

    def test_regional_salary_case_insensitive(self):
        result = calculate_affordability(800, region='LOMBARDIA')
        assert result['avg_salary'] == 2000.0

    def test_unknown_region_falls_back_to_national(self):
        result = calculate_affordability(800, region='NonExistent')
        assert result['avg_salary'] == 1650.0

    def test_zero_rent_does_not_crash(self):
        result = calculate_affordability(0)
        assert result['pct_of_avg_salary'] == 0.0
        assert result['is_affordable'] is True

    def test_region_missing_region_salary_data(self):
        """Calabria (lowest) should work."""
        result = calculate_affordability(500, region='Calabria')
        assert result['avg_salary'] == 1250.0

    def test_rent_price_float_conversion(self):
        result = calculate_affordability(800.75)
        assert result['rent_price'] == 800.75

    def test_affordability_color_green_for_highly_affordable(self):
        result = calculate_affordability(400)
        assert result['affordability_color'] == 'green'

    def test_affordability_color_lightgreen_for_affordable(self):
        result = calculate_affordability(600)
        assert result['affordability_color'] == 'lightgreen'

    def test_affordability_color_orange_for_moderate(self):
        result = calculate_affordability(750)
        assert result['affordability_color'] == 'orange'

    def test_affordability_color_red_for_challenging(self):
        result = calculate_affordability(1000)
        assert result['affordability_color'] == 'red'


# =============================================================================
# generate_prediction_report tests — covers FU-2
# =============================================================================

class TestGeneratePredictionReport:
    """Tests for generate_prediction_report — covers FU-2 (.get() fallback)."""

    def test_minimal_prediction_data(self):
        """Report handles minimal dict with .get() defaults (FU-2)."""
        report = generate_prediction_report(
            prediction_data={},
            similar_properties=[],
            affordability=calculate_affordability(800),
        )
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'PREDICTION SUMMARY' in report

    def test_full_prediction_data(self):
        """Report includes all sections with valid data."""
        prediction_data = {
            'address': 'Milano, Italy',
            'euro_est': 800.0,
            'confidence_score': 75.0,
            'lower_bound': 600.0,
            'upper_bound': 1000.0,
            'top_contributors': [
                {'feature': 'Area', 'contribution_euro': 100},
                {'feature': 'Location', 'contribution_euro': -50},
            ],
        }
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=[],
            affordability=calculate_affordability(800),
        )
        assert 'PREDICTION SUMMARY' in report
        assert '€800.00' in report
        assert '75.0%' in report
        assert 'Area' in report

    def test_report_includes_affordability(self):
        prediction_data = {'euro_est': 800, 'confidence_score': 50, 'lower_bound': 600, 'upper_bound': 1000, 'top_contributors': []}
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=[],
            affordability=calculate_affordability(800),
        )
        assert 'AFFORDABILITY ANALYSIS' in report
        assert '30% rule' in report

    def test_report_includes_similar_properties(self):
        prediction_data = {'euro_est': 800, 'confidence_score': 50, 'lower_bound': 600, 'upper_bound': 1000, 'top_contributors': []}
        similar = [
            {'neighborhood': 'Test', 'city': 'Milano', 'price': 750.0, 'rooms': 2, 'area': 80.0,
             'bathrooms': 1, 'energy_class': 'C', 'furnished': 'Yes', 'balcony': 'Yes', 'parking': 'No'}
        ]
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=similar,
            affordability=calculate_affordability(800),
        )
        assert 'SIMILAR PROPERTIES' in report
        assert 'Test, Milano' in report

    def test_report_includes_historical_stats(self):
        prediction_data = {'euro_est': 800, 'confidence_score': 50, 'lower_bound': 600, 'upper_bound': 1000, 'top_contributors': []}
        hist_stats = {
            'trend': 'rising',
            'price_change_pct': 5.2,
            'avg_price_overall': 750.0,
            'total_listings': 100,
        }
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=[],
            affordability=calculate_affordability(800),
            historical_stats=hist_stats,
        )
        assert 'HISTORICAL PRICE TRENDS' in report
        assert 'RISING' in report
        assert '+5.2%' in report

    def test_report_handles_missing_keys_gracefully(self):
        """No KeyError when optional sections missing (FU-2)."""
        report = generate_prediction_report(
            prediction_data={'address': 'Test'},
            similar_properties=[],
            affordability=calculate_affordability(800),
        )
        assert isinstance(report, str)

    def test_report_handles_none_historical_stats(self):
        prediction_data = {'euro_est': 800, 'confidence_score': 50, 'lower_bound': 600, 'upper_bound': 1000, 'top_contributors': []}
        # Should not crash with None historical_stats (FU-7 simplification)
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=[],
            affordability=calculate_affordability(800),
            historical_stats=None,
        )
        assert isinstance(report, str)

    def test_report_contains_disclaimer(self):
        prediction_data = {'euro_est': 800, 'confidence_score': 50, 'lower_bound': 600, 'upper_bound': 1000, 'top_contributors': []}
        report = generate_prediction_report(
            prediction_data=prediction_data,
            similar_properties=[],
            affordability=calculate_affordability(800),
        )
        assert 'DISCLAIMER' in report
