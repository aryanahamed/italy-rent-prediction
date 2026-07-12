"""Semantic tests for user-facing analysis helpers."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_utils import (
    calculate_affordability,
    find_similar_properties,
    generate_prediction_report,
    get_historical_price_trends,
)


def _properties(**overrides):
    data = {
        'city': ['Milano', 'Milano', 'Roma'],
        'neighborhood': ['Brera', 'Navigli', 'Centro'],
        'rooms': [2, 3, 2],
        'area': [80.0, 88.0, 80.0],
        'price': [1500.0, 1200.0, 900.0],
        'bathrooms': [1, 2, 1],
        'energy_class': ['C', 'B', 'D'],
        'furnished': [1, 0, 1],
        'balcony': [1, 0, 1],
        'parking_spots': [0, 1, 0],
        'condition': ['good', 'excellent', 'good'],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestFindSimilarProperties:
    def test_returns_same_city_only(self, monkeypatch):
        monkeypatch.setattr('feature_utils.load_rental_data', _properties)
        results = find_similar_properties('milano', rooms=2, area=80, top_n=5)
        assert results
        assert {item['city'] for item in results} == {'Milano'}
        assert all(item['city_matched'] is True for item in results)

    def test_does_not_fallback_to_another_city(self, monkeypatch):
        monkeypatch.setattr('feature_utils.load_rental_data', _properties)
        assert find_similar_properties('Torino', rooms=2, area=80) == []

    def test_prediction_price_is_not_used_for_selection(self, monkeypatch):
        monkeypatch.setattr('feature_utils.load_rental_data', _properties)
        low = find_similar_properties('Milano', rooms=2, area=80, price=1)
        high = find_similar_properties('Milano', rooms=2, area=80, price=100_000)
        assert [item['neighborhood'] for item in low] == [item['neighborhood'] for item in high]

    def test_high_valid_asking_rent_is_not_deleted_as_global_outlier(self, monkeypatch):
        data = _properties(price=[1500.0, 20_000.0, 900.0])
        monkeypatch.setattr('feature_utils.load_rental_data', lambda: data)
        results = find_similar_properties('Milano', rooms=2, area=80, top_n=5)
        assert 20_000.0 in {item['price'] for item in results}

    def test_exact_duplicates_do_not_fill_multiple_slots(self, monkeypatch):
        duplicate = pd.concat([_properties().iloc[[0]], _properties().iloc[[0]]], ignore_index=True)
        monkeypatch.setattr('feature_utils.load_rental_data', lambda: duplicate)
        assert len(find_similar_properties('Milano', rooms=2, area=80)) == 1

    def test_score_is_fixed_and_interpretable(self, monkeypatch):
        monkeypatch.setattr('feature_utils.load_rental_data', _properties)
        result = find_similar_properties('Milano', rooms=2, area=80, top_n=1)[0]
        assert result['similarity_score'] == pytest.approx(1.0)

    def test_invalid_search_returns_empty(self, monkeypatch):
        monkeypatch.setattr('feature_utils.load_rental_data', _properties)
        assert find_similar_properties('', rooms=2, area=80) == []
        assert find_similar_properties('Milano', rooms=0, area=80) == []
        assert find_similar_properties('Milano', rooms=2, area=0) == []

    def test_data_failure_is_not_disguised_as_no_matches(self, monkeypatch):
        def fail():
            raise OSError('missing dataset')
        monkeypatch.setattr('feature_utils.load_rental_data', fail)
        with pytest.raises(RuntimeError, match='missing dataset'):
            find_similar_properties('Milano', rooms=2, area=80)

    def test_missing_amenity_is_not_presented_as_no(self, monkeypatch):
        data = _properties().iloc[[0]].copy()
        for column in ['furnished', 'balcony', 'parking_spots']:
            data[column] = pd.Series([float('nan')], index=data.index, dtype=float)
        monkeypatch.setattr('feature_utils.load_rental_data', lambda: data)
        result = find_similar_properties('Milano', rooms=2, area=80)[0]
        assert result['furnished'] == 'Unknown'
        assert result['balcony'] == 'Unknown'
        assert result['parking'] == 'Unknown'


class TestCalculateAffordability:
    def test_without_income_makes_no_verdict(self):
        result = calculate_affordability(900)
        assert result['monthly_household_income'] is None
        assert result['pct_of_income'] is None
        assert result['is_within_reference'] is None
        assert result['ratio_level'] == 'Income not provided'
        assert result['required_income'] == pytest.approx(3000)

    def test_uses_submitted_household_income(self):
        result = calculate_affordability(900, monthly_household_income=3000)
        assert result['pct_of_income'] == pytest.approx(30)
        assert result['is_within_reference'] is True

    def test_above_reference_is_not_called_affordable(self):
        result = calculate_affordability(1200, monthly_household_income=3000)
        assert result['pct_of_income'] == pytest.approx(40)
        assert result['is_within_reference'] is False
        assert result['ratio_level'] == 'Above 30% reference'

    def test_high_ratio_band(self):
        result = calculate_affordability(1500, monthly_household_income=3000)
        assert result['ratio_band'] == 'high_ratio'

    @pytest.mark.parametrize('rent', [-1, float('nan'), float('inf')])
    def test_invalid_rent_rejected(self, rent):
        with pytest.raises(ValueError, match='rent_price'):
            calculate_affordability(rent)

    @pytest.mark.parametrize('income', [0, -1, float('nan'), float('inf')])
    def test_invalid_provided_income_rejected(self, income):
        with pytest.raises(ValueError, match='monthly_household_income'):
            calculate_affordability(900, income)


def test_historical_trends_refuse_snapshot_data():
    with pytest.raises(RuntimeError, match='not a longitudinal dataset'):
        get_historical_price_trends('Milano')


def _prediction(**overrides):
    prediction = {
        'address': 'Brera, Milano, Italia',
        'euro_est': 1500.0,
        'lower_bound': 1400.0,
        'upper_bound': 1620.0,
        'stability_score': 85.0,
        'data_as_of': '2023-12-07',
        'inputs': {'rooms': 2, 'area_m2': 80, 'monthly_household_income': 4000},
        'top_contributors': [
            {'feature': 'Balcony', 'contribution_euro': 90.0},
            {'feature': 'Parking', 'contribution_euro': -40.0},
        ],
    }
    prediction.update(overrides)
    return prediction


class TestGeneratePredictionReport:
    def test_uses_honest_uncertainty_language(self):
        report = generate_prediction_report(
            _prediction(), [], calculate_affordability(1500, 4000)
        )
        assert 'Input Stability Score: 85.0/100' in report
        assert 'Input Perturbation Range' in report
        assert 'not a prediction interval, probability, or accuracy guarantee' in report
        assert '95%' not in report
        assert 'Confidence Score' not in report

    def test_includes_immutable_submitted_snapshot(self):
        report = generate_prediction_report(
            _prediction(), [], calculate_affordability(1500, 4000)
        )
        assert 'SUBMITTED PROPERTY' in report
        assert 'Rooms: 2' in report
        assert 'Area M2: 80' in report
        assert 'Data through: 2023-12-07' in report

    def test_sensitivities_are_non_causal(self):
        report = generate_prediction_report(
            _prediction(), [], calculate_affordability(1500, 4000)
        )
        assert 'estimate is higher by €90.00' in report
        assert 'estimate is lower by €40.00' in report
        assert 'not causal effects' in report

    def test_no_income_produces_no_affordability_verdict(self):
        report = generate_prediction_report(
            _prediction(), [], calculate_affordability(1500)
        )
        assert 'Household income was not provided; no affordability verdict is made.' in report

    def test_legacy_historical_stats_are_ignored(self):
        report = generate_prediction_report(
            _prediction(), [], calculate_affordability(1500),
            historical_stats={'trend': 'rising', 'price_change_pct': 99},
        )
        assert 'HISTORICAL PRICE TRENDS' not in report
        assert 'rising' not in report.lower()

    def test_comparables_are_labeled_and_rendered(self):
        comparable = {
            'neighborhood': 'Brera', 'city': 'Milano', 'price': 1450,
            'rooms': 2, 'area': 78, 'bathrooms': 1, 'energy_class': 'C',
            'furnished': 'Yes', 'balcony': 'Yes', 'parking': 'No',
        }
        report = generate_prediction_report(
            _prediction(), [comparable], calculate_affordability(1500)
        )
        assert 'SIMILAR PROPERTIES (1 found)' in report
        assert 'Brera, Milano' in report
