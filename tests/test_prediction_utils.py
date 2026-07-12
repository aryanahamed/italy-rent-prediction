"""Tests for prediction_utils.py — PredictionAnalyzer and formatting functions."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from prediction_utils import (
    PredictionAnalyzer,
    format_confidence_level,
    format_stability_level,
    format_contribution_text,
)


# =============================================================================
# Helper: mock model factory
# =============================================================================

class MockXGBoostModel:
    """Simulates an XGBRegressor with basic sklearn-compatible interface."""
    def __init__(self, n_features=36):
        self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]
        self.feature_importances_ = np.random.dirichlet(np.ones(n_features))
        self.n_features_in_ = n_features

    def predict(self, X):
        # Simple linear prediction in log space
        return np.array([np.log1p(np.sum(X) * 50 + 500)])


class MockRandomForestModel:
    """Simulates a RandomForestRegressor with estimators."""
    def __init__(self, n_features=36, n_estimators=10):
        self.feature_names_in_ = [f'feature_{i}' for i in range(n_features)]
        self.feature_importances_ = np.random.dirichlet(np.ones(n_features))
        self.n_features_in_ = n_features
        self.estimators_ = [MockTree() for _ in range(n_estimators)]

    def predict(self, X):
        return np.array([np.log1p(np.sum(X) * 50 + 500)])


class MockTree:
    """Simulates a single decision tree estimator."""
    def predict(self, X):
        return np.array([np.log1p(np.sum(X) * 50 + 500)])


class MockModelNoFeatureNames:
    """Model without feature_names_in_ attribute — tests PU-3 fix."""
    def __init__(self, n_features=36):
        self.feature_importances_ = np.random.dirichlet(np.ones(n_features))
        self.n_features_in_ = n_features

    def predict(self, X):
        return np.array([np.log1p(np.sum(X) * 50 + 500)])


# =============================================================================
# PredictionAnalyzer tests
# =============================================================================

class TestPredictionAnalyzerInit:
    """Tests for constructor — covers PU-3."""

    def test_init_with_feature_names(self):
        """Analyzer reads feature_names_in_ when available."""
        model = MockXGBoostModel(n_features=36)
        analyzer = PredictionAnalyzer(model)
        assert len(analyzer.feature_names) == 36
        assert analyzer.feature_names[0] == 'feature_0'

    def test_init_without_feature_names(self):
        """Analyzer falls back to generated names when feature_names_in_ missing (PU-3)."""
        model = MockModelNoFeatureNames(n_features=10)
        analyzer = PredictionAnalyzer(model)
        assert len(analyzer.feature_names) == 10
        assert analyzer.feature_names[0] == 'feature_0'

    def test_init_detects_xgboost(self):
        model = MockXGBoostModel()
        analyzer = PredictionAnalyzer(model)
        assert analyzer.is_xgboost is True
        assert analyzer.is_random_forest is False

    def test_init_detects_random_forest(self):
        model = MockRandomForestModel()
        analyzer = PredictionAnalyzer(model)
        assert analyzer.is_xgboost is False
        assert analyzer.is_random_forest is True

    def test_init_stores_feature_importances(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        assert len(analyzer.feature_importances) == 5

    def test_init_accepts_feature_medians(self):
        """feature_medians parameter is stored (PU-6)."""
        model = MockXGBoostModel(n_features=5)
        medians = np.array([100, 50, 30, 20, 10])
        analyzer = PredictionAnalyzer(model, feature_medians=medians)
        assert analyzer.feature_medians is not None
        assert analyzer.feature_medians[0] == 100


class TestCalculateConfidenceScore:
    """Tests for calculate_confidence_score — covers PU-2, PU-4, PU-5."""

    def test_xgboost_returns_tuple(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        result = analyzer.calculate_confidence_score(features)
        assert len(result) == 4
        pred, lower, upper, conf = result
        assert isinstance(pred, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert isinstance(conf, float)
        assert 0 <= conf <= 100

    def test_xgboost_prediction_in_range(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        pred, lower, upper, conf = analyzer.calculate_confidence_score(features)
        assert lower <= pred <= upper

    def test_random_forest_returns_tuple(self):
        model = MockRandomForestModel(n_features=5, n_estimators=10)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        result = analyzer.calculate_confidence_score(features)
        assert len(result) == 4
        pred, lower, upper, conf = result
        assert 0 <= conf <= 100

    def test_random_forest_prediction_in_range(self):
        model = MockRandomForestModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        pred, lower, upper, conf = analyzer.calculate_confidence_score(features)
        assert lower <= pred <= upper

    def test_confidence_is_deterministic(self):
        """Same input produces same confidence (PU-5 fix)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        _, _, _, conf1 = analyzer.calculate_confidence_score(features)
        _, _, _, conf2 = analyzer.calculate_confidence_score(features)
        assert conf1 == pytest.approx(conf2, abs=1e-6)

    def test_confidence_non_nan_for_edge_inputs(self):
        """No NaN returned for extreme inputs (PU-2 fix)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        # Test near-zero prediction scenario
        features = np.array([[0, 0, 0, 0, 0]])
        _, _, _, conf = analyzer.calculate_confidence_score(features)
        assert not np.isnan(conf), f"Confidence should not be NaN, got {conf}"
        assert 0 <= conf <= 100

    def test_confidence_non_nan_for_large_values(self):
        """No NaN for large inputs (PU-2 overflow protection)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1e6, 1e6, 1e6, 1e6, 1e6]])
        _, _, _, conf = analyzer.calculate_confidence_score(features)
        assert not np.isnan(conf)
        assert 0 <= conf <= 100

    def test_handles_multi_row_input(self):
        """Uses only first row if multi-row passed (PU-4)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1], [0, 0, 0, 0, 0]])
        result = analyzer.calculate_confidence_score(features)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_handles_1d_input(self):
        """Wraps 1D array automatically (PU-4)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([1, 2, 3, 0, 1])
        result = analyzer.calculate_confidence_score(features)
        assert len(result) == 4

    def test_unknown_model_type(self):
        """Uses conservative estimate for unknown model types."""
        class MockUnknownModel:
            def __init__(self):
                self.feature_names_in_ = ['a', 'b']
                self.feature_importances_ = np.array([0.5, 0.5])
                self.n_features_in_ = 2
            def predict(self, X):
                return np.array([7.0])

        model = MockUnknownModel()
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2]])
        pred, lower, upper, conf = analyzer.calculate_confidence_score(features)
        assert 0 <= conf <= 100
        assert lower <= pred <= upper

    def test_supported_input_changes_recompute_derived_ratios(self):
        class RecordingModel:
            feature_names_in_ = np.array([
                'bathrooms', 'rooms', 'area', 'rooms_per_area', 'baths_per_room'
            ])
            feature_importances_ = np.ones(5) / 5
            n_features_in_ = 5
            def __init__(self):
                self.seen = []
            def predict(self, X):
                self.seen.extend(np.atleast_2d(X).copy())
                return np.log1p(np.atleast_2d(X)[:, 2] * 10)

        model = RecordingModel()
        features = np.array([[1, 2, 80, 2 / 80, 1 / 2]], dtype=float)
        PredictionAnalyzer(model).calculate_confidence_score(features)
        for row in model.seen:
            assert row[3] == pytest.approx(row[1] / row[2])
            assert row[4] == pytest.approx(row[0] / row[1])

    def test_stability_width_is_calculated_in_euros(self):
        class AreaModel:
            feature_names_in_ = np.array(['area'])
            feature_importances_ = np.array([1.0])
            n_features_in_ = 1
            def predict(self, X):
                return np.log1p(np.atleast_2d(X)[:, 0] * 10)

        _, _, _, stability = PredictionAnalyzer(AreaModel()).calculate_confidence_score(
            np.array([[100.0]])
        )
        assert stability == pytest.approx(90.0)


class TestGetFeatureContributions:
    """Tests for get_feature_contributions — covers PU-1, PU-4, PU-6, PU-7."""

    def test_returns_list_of_dicts(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model, feature_medians=np.zeros(5))
        features = np.array([[1, 2, 3, 0, 1]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        assert isinstance(contribs, list)
        assert len(contribs) > 0
        for c in contribs:
            assert 'feature' in c
            assert 'contribution_euro' in c
            assert 'importance' in c

    def test_handles_multi_row_input(self):
        """Uses only first row (PU-4)."""
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model, feature_medians=np.zeros(5))
        features = np.array([[1, 2, 3, 0, 1], [0, 0, 0, 0, 0]])
        pred = model.predict(features[:1])[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        assert len(contribs) > 0

    def test_handles_1d_input(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model, feature_medians=np.zeros(5))
        features = np.array([1, 2, 3, 0, 1])
        pred = model.predict(features.reshape(1, -1))[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        assert len(contribs) > 0

    def test_sorted_by_absolute_contribution(self):
        model = MockXGBoostModel(n_features=5)
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        abs_vals = [abs(c['contribution_euro']) for c in contribs]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_skips_low_importance_features(self):
        model = MockXGBoostModel(n_features=5)
        # Force one feature to near-zero importance
        model.feature_importances_[0] = 0.0001
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 2, 3, 0, 1]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        # Feature 0 should be skipped
        raw_names = [c['raw_name'] for c in contribs]
        assert 'feature_0' not in raw_names

    def test_binary_feature_baseline(self):
        """Binary features compare with a clear “not present” reference."""
        model = MockXGBoostModel(n_features=2)
        model.feature_names_in_ = ['parking_spots', 'balcony']
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 0]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        for c in contribs:
            assert c['baseline_value'] == 0.0

    def test_count_feature_with_value_one_gets_zero_baseline(self):
        """Continuous/count features are omitted without training medians."""
        model = MockXGBoostModel(n_features=2)
        model.feature_names_in_ = ['bathrooms', 'rooms']
        analyzer = PredictionAnalyzer(model)
        # bathrooms=1, rooms=1 — both count features, neither in BINARY_FEATURE_NAMES
        features = np.array([[1, 1]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        assert contribs == []

    def test_mixed_binary_and_count_features(self):
        """Binary references use zero and continuous references use supplied medians."""
        model = MockXGBoostModel(n_features=4)
        model.feature_names_in_ = ['parking_spots', 'furnished', 'bathrooms', 'rooms']
        analyzer = PredictionAnalyzer(model, feature_medians=np.array([0, 0, 2, 3]))
        features = np.array([[1, 1, 1, 1]])  # All values are 1
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        by_name = {c['raw_name']: c for c in contribs}

        # Binary features
        assert by_name['parking_spots']['baseline_value'] == 0.0
        assert by_name['furnished']['baseline_value'] == 0.0

        # Count features (value 1, but not truly binary)
        assert by_name['bathrooms']['baseline_value'] == 2
        assert by_name['rooms']['baseline_value'] == 3

    def test_binary_feature_with_underscored_name(self):
        """Underscore variants of binary feature names are also recognized."""
        model = MockXGBoostModel(n_features=2)
        model.feature_names_in_ = ['central_heating', 'external_exposure']
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1, 0]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        for c in contribs:
            assert c['baseline_value'] == 0.0

    def test_derived_interaction_is_omitted_without_training_reference(self):
        model = MockXGBoostModel(n_features=1)
        model.feature_names_in_ = ['Furnished and Central Heating']
        analyzer = PredictionAnalyzer(model)
        features = np.array([[1]])
        assert analyzer.get_feature_contributions(features, model.predict(features)[0]) == []

    def test_rf_returns_valid_contributions(self):
        """RF path works without broken tree-threshold logic (PU-1)."""
        model = MockRandomForestModel(n_features=5, n_estimators=5)
        analyzer = PredictionAnalyzer(model, feature_medians=np.zeros(5))
        features = np.array([[1, 2, 3, 0, 1]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        assert len(contribs) > 0
        assert all(isinstance(c['contribution_euro'], float) for c in contribs)

    def test_feature_medians_used_when_provided(self):
        """Provided feature medians are used as baseline for non-binary features (PU-6)."""
        model = MockXGBoostModel(n_features=3)
        medians = np.array([10.0, 20.0, 30.0])
        analyzer = PredictionAnalyzer(model, feature_medians=medians)
        # Use non-binary values (not 0 or 1)
        features = np.array([[5, 15, 25]])
        pred = model.predict(features)[0]
        contribs = analyzer.get_feature_contributions(features, pred)
        for c in contribs:
            idx = int(c['raw_name'].split('_')[1])
            # Non-binary features should use the provided median
            assert c['baseline_value'] == medians[idx], f"Feature {idx}: expected {medians[idx]}, got {c['baseline_value']}"


class TestGetTopContributors:
    """Tests for get_top_contributors — covers PU-9."""

    def test_returns_top_n(self):
        contributions = [
            {'feature': 'A', 'raw_name': 'a', 'contribution_euro': 100, 'importance': 0.5},
            {'feature': 'B', 'raw_name': 'b', 'contribution_euro': 50, 'importance': 0.3},
            {'feature': 'C', 'raw_name': 'c', 'contribution_euro': 25, 'importance': 0.2},
        ]
        top = PredictionAnalyzer.get_top_contributors(None, contributions, top_n=2)
        assert len(top) == 2

    def test_combines_location_features(self):
        """Location features combined into single entry (PU-9)."""
        contributions = [
            {'feature': 'Rooms', 'raw_name': 'rooms', 'contribution_euro': 100, 'importance': 0.4, 'contribution_log': 0.1},
            {'feature': 'Location (Latitude)', 'raw_name': 'latitude', 'contribution_euro': 50, 'importance': 0.2, 'contribution_log': 0.05},
            {'feature': 'Location (Longitude)', 'raw_name': 'longitude', 'contribution_euro': -30, 'importance': 0.15, 'contribution_log': -0.03},
        ]
        top = PredictionAnalyzer.get_top_contributors(None, contributions, top_n=5)
        # Location should be combined
        location_entries = [c for c in top if c['raw_name'] == 'location_combined']
        assert len(location_entries) == 1
        # Combined contribution should be sum
        assert location_entries[0]['contribution_euro'] == 20  # 50 + (-30)

    def test_prefix_matching_catches_all_location_features(self):
        """Uses startswith for location detection (PU-9)."""
        contributions = [
            {'feature': 'Test', 'raw_name': 'latitude_test', 'contribution_euro': 10, 'importance': 0.1, 'contribution_log': 0.01},
            {'feature': 'Test2', 'raw_name': 'longitude_test', 'contribution_euro': 20, 'importance': 0.2, 'contribution_log': 0.02},
        ]
        top = PredictionAnalyzer.get_top_contributors(None, contributions, top_n=5)
        combined = [c for c in top if c['raw_name'] == 'location_combined']
        assert len(combined) == 1


class TestFormatFeatureName:
    """Tests for _format_feature_name — covers PU-8."""

    def test_known_feature_mapped(self):
        model = MockXGBoostModel(n_features=1)
        analyzer = PredictionAnalyzer(model)
        assert analyzer._format_feature_name('bathrooms') == 'Bathrooms'
        assert analyzer._format_feature_name('area') == 'Area'
        assert analyzer._format_feature_name('balcony') == 'Balcony'

    def test_unknown_feature_uses_title(self):
        """Unknown features use replace('_', ' ').title() (PU-8)."""
        model = MockXGBoostModel(n_features=1)
        analyzer = PredictionAnalyzer(model)
        result = analyzer._format_feature_name('custom_feature_name')
        assert result == 'Custom Feature Name'

    def test_underscore_handled_correctly(self):
        """Underscores replaced with spaces before title() (PU-8)."""
        model = MockXGBoostModel(n_features=1)
        analyzer = PredictionAnalyzer(model)
        result = analyzer._format_feature_name('some_deep_feature')
        assert '_' not in result


# =============================================================================
# format_confidence_level tests — covers PU-10
# =============================================================================

class TestFormatConfidenceLevel:
    def test_very_high(self):
        assert format_confidence_level(95) == 'Very High'
        assert format_confidence_level(90) == 'Very High'

    def test_high(self):
        assert format_confidence_level(85) == 'High'
        assert format_confidence_level(75) == 'High'

    def test_moderate(self):
        assert format_confidence_level(65) == 'Moderate'
        assert format_confidence_level(60) == 'Moderate'

    def test_low(self):
        assert format_confidence_level(50) == 'Low'
        assert format_confidence_level(40) == 'Low'

    def test_very_low(self):
        assert format_confidence_level(30) == 'Very Low'
        assert format_confidence_level(0) == 'Very Low'

    def test_boundary_values(self):
        assert format_confidence_level(100) == 'Very High'
        assert format_confidence_level(89.9) == 'High'
        assert format_confidence_level(74.9) == 'Moderate'
        assert format_confidence_level(59.9) == 'Low'


class TestFormatStabilityLevel:
    def test_uses_diagnostic_not_certainty_language(self):
        assert format_stability_level(95) == 'Very Stable'
        assert format_stability_level(80) == 'Stable'
        assert format_stability_level(65) == 'Moderately Stable'
        assert format_stability_level(45) == 'Sensitive'
        assert format_stability_level(10) == 'Very Sensitive'


# =============================================================================
# format_contribution_text tests — covers PU-11
# =============================================================================

class TestFormatContributionText:
    def test_no_impact(self):
        """Very small contributions get 'no impact' (PU-11)."""
        result = format_contribution_text({'feature': 'Test', 'contribution_euro': 0.005})
        assert 'no impact' in result

    def test_minimal_impact(self):
        result = format_contribution_text({'feature': 'Test', 'contribution_euro': 0.5})
        assert 'minimal impact' in result

    def test_adds_rent(self):
        result = format_contribution_text({'feature': 'Balcony', 'contribution_euro': 150})
        assert 'higher vs reference' in result
        assert '€150' in result

    def test_reduces_rent(self):
        result = format_contribution_text({'feature': 'Rooms', 'contribution_euro': -75})
        assert 'lower vs reference' in result
        assert '€75' in result

    def test_exact_zero(self):
        result = format_contribution_text({'feature': 'Test', 'contribution_euro': 0.0})
        assert 'no impact' in result
