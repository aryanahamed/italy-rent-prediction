"""Tests for the serialized-model runtime and transformation contract."""

import json
from pathlib import Path

import numpy as np
import pytest

from config import MIN_RENT
from model_contract import (
    ModelContractError,
    calculate_baths_per_room,
    calculate_rooms_per_area,
    load_model_metadata,
    log_rent_to_euro,
    validate_model_features,
    validate_xgboost_runtime,
)


def test_metadata_records_artifact_runtime_and_target_transform():
    metadata = load_model_metadata()
    assert metadata['framework'] == 'xgboost'
    assert metadata['framework_version'] == '3.1.1'
    assert metadata['target_transform'] == 'log1p'
    assert metadata['feature_count'] == 36


def test_requirements_install_a_compatible_xgboost_runtime():
    requirements = Path('requirements.txt').read_text(encoding='utf-8')
    assert 'xgboost>=3.1.1,<4.0.0' in requirements


def test_runtime_rejects_xgboost_2_for_xgboost_3_artifact():
    with pytest.raises(ModelContractError, match='requires XGBoost 3.1.1'):
        validate_xgboost_runtime({'framework_version': '3.1.1'}, '2.1.4')


@pytest.mark.parametrize('runtime', ['3.1.1', '3.2.0'])
def test_runtime_accepts_compatible_xgboost_3(runtime):
    validate_xgboost_runtime({'framework_version': '3.1.1'}, runtime)


def test_runtime_rejects_unreviewed_next_major():
    with pytest.raises(ModelContractError):
        validate_xgboost_runtime({'framework_version': '3.1.1'}, '4.0.0')


def test_derived_features_match_training_notebook():
    assert calculate_rooms_per_area(2, 50) == pytest.approx(2 / 51)
    assert calculate_baths_per_room(1, 2) == pytest.approx(1 / 3)


def test_log_target_conversion():
    assert log_rent_to_euro(np.log1p(710), enforce_minimum=True) == pytest.approx(710)


def test_near_zero_prediction_is_rejected_instead_of_displayed_as_success():
    with pytest.raises(ModelContractError, match='below the supported minimum'):
        log_rent_to_euro(np.log1p(MIN_RENT - 1), enforce_minimum=True)


def test_feature_order_mismatch_is_rejected(tmp_path):
    expected_path = tmp_path / 'features.json'
    expected_path.write_text(json.dumps(['rooms', 'area']), encoding='utf-8')

    class Model:
        feature_names_in_ = ['area', 'rooms']

    with pytest.raises(ModelContractError, match='feature names/order'):
        validate_model_features(Model(), expected_path)
