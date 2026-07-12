"""Tests for query-bound location state and immutable result snapshots."""

from app_state import (
    build_prediction_snapshot,
    get_current_location,
    store_resolved_location,
    sync_location_query,
)


def _store(state, query='Brera Milano'):
    return store_resolved_location(
        state,
        query=query,
        display_label='Brera, Milano, Italia',
        latitude=45.471,
        longitude=9.188,
        region='Lombardia',
        city='Milano',
        neighborhood='Brera',
        region_coordinates=(45.5, 9.5),
        city_coordinates=(45.464, 9.19),
        country_code='IT',
        place_id='123',
    )


def test_query_change_clears_resolved_location_and_prediction():
    state = {'location_query_normalized': 'brera milano', 'prediction_results': {'old': True}}
    _store(state)
    assert sync_location_query(state, 'Navigli Milano') is True
    assert 'resolved_location' not in state
    assert 'latitude' not in state
    assert state['prediction_results'] is None


def test_whitespace_and_case_do_not_invalidate_location():
    state = {'location_query_normalized': 'brera milano'}
    _store(state)
    assert sync_location_query(state, '  BRERA   milano ') is False
    assert get_current_location(state, 'brera milano')['city'] == 'Milano'


def test_location_must_match_current_query():
    state = {}
    _store(state)
    assert get_current_location(state, 'Roma') is None


def test_non_italian_location_is_rejected():
    state = {}
    _store(state)
    state['resolved_location']['country_code'] = 'FR'
    assert get_current_location(state, 'Brera Milano') is None


def test_selecting_a_different_result_invalidates_old_prediction():
    state = {'prediction_results': {'old': True}}
    _store(state)
    state['prediction_results'] = {'old': True}
    store_resolved_location(
        state,
        query='Brera Milano', display_label='Milano, Italia',
        latitude=45.464, longitude=9.190, region='Lombardia', city='Milano',
        neighborhood='', region_coordinates=(45.5, 9.5),
        city_coordinates=(45.464, 9.190), country_code='IT', place_id='456',
    )
    assert state['prediction_results'] is None


def test_prediction_snapshot_is_deep_copied():
    location = _store({})
    inputs = {'rooms': 2, 'amenities': ['Balcony']}
    sensitivities = [{'feature': 'Balcony', 'contribution_euro': 50}]
    snapshot = build_prediction_snapshot(
        location=location,
        inputs=inputs,
        estimate=1500,
        perturbation_lower=1400,
        perturbation_upper=1600,
        stability_score=87,
        sensitivity_items=sensitivities,
        data_as_of='2023-12-07',
    )
    inputs['rooms'] = 5
    inputs['amenities'].append('Pool')
    sensitivities[0]['contribution_euro'] = 999
    assert snapshot['inputs'] == {'rooms': 2, 'amenities': ['Balcony']}
    assert snapshot['sensitivity_items'][0]['contribution_euro'] == 50
    assert snapshot['address'] == 'Brera, Milano, Italia'
