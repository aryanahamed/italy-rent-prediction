"""Pure helpers for keeping location and prediction state internally consistent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping


LOCATION_KEYS = (
    "latitude",
    "longitude",
    "latitude_region",
    "longitude_region",
    "latitude_city",
    "longitude_city",
    "latitude_neighborhood",
    "longitude_neighborhood",
    "region",
    "city",
    "neighborhood",
    "resolved_location",
    "last_map_address",
    "selected_location",
)


def normalize_query(query: str | None) -> str:
    """Normalize a location query for state comparisons."""
    return " ".join((query or "").strip().casefold().split())


def clear_location_state(
    state: MutableMapping[str, Any], *, clear_prediction: bool = True
) -> None:
    """Remove every value derived from a previously resolved location."""
    for key in LOCATION_KEYS:
        state.pop(key, None)
    if clear_prediction:
        state["prediction_results"] = None


def sync_location_query(state: MutableMapping[str, Any], query: str | None) -> bool:
    """Clear derived state when the user's location query changes.

    Returns ``True`` when a change was detected.
    """
    normalized = normalize_query(query)
    previous = state.get("location_query_normalized")
    changed = previous is not None and previous != normalized
    if changed:
        clear_location_state(state)
    state["location_query_normalized"] = normalized
    return changed


def store_resolved_location(
    state: MutableMapping[str, Any],
    *,
    query: str,
    display_label: str,
    latitude: float,
    longitude: float,
    region: str,
    city: str,
    neighborhood: str,
    region_coordinates: tuple[float, float],
    city_coordinates: tuple[float, float],
    country_code: str = "IT",
    place_id: str | None = None,
) -> dict[str, Any]:
    """Store one canonical, query-bound location and compatibility keys."""
    previous = state.get("resolved_location")
    location = {
        "query": query.strip(),
        "query_normalized": normalize_query(query),
        "display_label": display_label,
        "place_id": place_id,
        "country_code": country_code.upper(),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "region": region or "",
        "city": city or "",
        "neighborhood": neighborhood or "",
        "latitude_region": float(region_coordinates[0]),
        "longitude_region": float(region_coordinates[1]),
        "latitude_city": float(city_coordinates[0]),
        "longitude_city": float(city_coordinates[1]),
        "latitude_neighborhood": float(latitude),
        "longitude_neighborhood": float(longitude),
    }
    if isinstance(previous, Mapping):
        previous_identity = (
            previous.get("place_id"), previous.get("latitude"), previous.get("longitude")
        )
        new_identity = (location["place_id"], location["latitude"], location["longitude"])
        if previous_identity != new_identity:
            state["prediction_results"] = None
    state["resolved_location"] = location
    for key in LOCATION_KEYS:
        if key in location:
            state[key] = location[key]
    return deepcopy(location)


def get_current_location(
    state: Mapping[str, Any], query: str | None
) -> dict[str, Any] | None:
    """Return the resolved location only when it belongs to the current query."""
    location = state.get("resolved_location")
    if not isinstance(location, Mapping):
        return None
    if location.get("query_normalized") != normalize_query(query):
        return None
    if location.get("country_code") != "IT":
        return None
    return deepcopy(dict(location))


def build_prediction_snapshot(
    *,
    location: Mapping[str, Any],
    inputs: Mapping[str, Any],
    estimate: float,
    perturbation_lower: float,
    perturbation_upper: float,
    stability_score: float,
    sensitivity_items: list[dict[str, Any]],
    data_as_of: str,
) -> dict[str, Any]:
    """Build the immutable object used by every result section and report."""
    return {
        "location": deepcopy(dict(location)),
        "inputs": deepcopy(dict(inputs)),
        "address": str(location.get("display_label", "Unknown")),
        "euro_est": float(estimate),
        "lower_bound": float(perturbation_lower),
        "upper_bound": float(perturbation_upper),
        "confidence_score": float(stability_score),  # compatibility key
        "stability_score": float(stability_score),
        "top_contributors": deepcopy(sensitivity_items),  # compatibility key
        "sensitivity_items": deepcopy(sensitivity_items),
        "data_as_of": data_as_of,
    }
