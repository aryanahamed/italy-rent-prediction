"""Deployed model runtime and feature-transformation contract."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config import MIN_RENT, PROJECT_ROOT


MODEL_METADATA_PATH = PROJECT_ROOT / 'rent_prediction_model' / 'model_metadata.json'
FEATURE_NAMES_PATH = PROJECT_ROOT / 'rent_prediction_model' / 'feature_names.json'


class ModelContractError(RuntimeError):
    """Raised when runtime code cannot safely interpret the model artifact."""


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r'\d+', value)
    if not numbers:
        raise ModelContractError(f"Invalid framework version: {value!r}")
    return tuple(int(number) for number in numbers[:3])


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as metadata_file:
        return json.load(metadata_file)


def validate_xgboost_runtime(
    metadata: Mapping[str, Any], installed_version: str | None = None
) -> None:
    """Reject runtimes older than the one that serialized the artifact."""
    required = str(metadata['framework_version'])
    try:
        installed = installed_version or version('xgboost')
    except PackageNotFoundError as exc:
        raise ModelContractError("XGBoost is not installed.") from exc
    required_tuple = _version_tuple(required)
    installed_tuple = _version_tuple(installed)
    if installed_tuple[0] != required_tuple[0] or installed_tuple < required_tuple:
        raise ModelContractError(
            f"Model requires XGBoost {required} through <4.0; installed version is {installed}."
        )


def validate_model_features(model, feature_names_path: Path = FEATURE_NAMES_PATH) -> None:
    with feature_names_path.open('r', encoding='utf-8') as feature_file:
        expected = json.load(feature_file)
    actual = [str(name) for name in getattr(model, 'feature_names_in_', [])]
    if actual != expected:
        raise ModelContractError(
            "Model feature names/order do not match rent_prediction_model/feature_names.json."
        )


def calculate_rooms_per_area(rooms: float, area: float) -> float:
    """Match the exact derived feature used during model training."""
    return float(rooms) / (float(area) + 1.0)


def calculate_baths_per_room(bathrooms: float, rooms: float) -> float:
    """Match the exact derived feature used during model training."""
    return float(bathrooms) / (float(rooms) + 1.0)


def log_rent_to_euro(log_prediction: float, *, enforce_minimum: bool = False) -> float:
    """Convert the model's log1p target and reject unusable base estimates."""
    log_prediction = float(log_prediction)
    if not np.isfinite(log_prediction):
        raise ModelContractError("Model returned a non-finite prediction.")
    euro_prediction = float(np.expm1(log_prediction))
    if enforce_minimum and euro_prediction < MIN_RENT:
        raise ModelContractError(
            f"Model returned €{euro_prediction:.2f}, below the supported minimum of €{MIN_RENT:.0f}. "
            "The model/runtime contract may be incompatible."
        )
    return euro_prediction
