"""Regression tests for the hosted Photon request contract."""

import pytest

from geocoding import build_photon_params


def test_photon_params_restrict_results_to_italy_without_unsupported_language():
    params = build_photon_params('  venezia,   venezia  ', limit=5)
    assert params == {
        'q': 'venezia, venezia',
        'limit': 5,
        'countrycode': 'IT',
    }
    assert 'lang' not in params


@pytest.mark.parametrize('query', ['', '   ', None])
def test_photon_params_reject_empty_query(query):
    with pytest.raises(ValueError, match='query must not be empty'):
        build_photon_params(query)


def test_photon_params_reject_non_positive_limit():
    with pytest.raises(ValueError, match='limit must be positive'):
        build_photon_params('Venezia', limit=0)
