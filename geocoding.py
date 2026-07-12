"""Photon request-contract helpers."""

from config import PHOTON_COUNTRY_CODE


def build_photon_params(query: str, *, limit: int = 5) -> dict[str, str | int]:
    """Build parameters accepted by the hosted Photon API.

    The public instance does not support Italian as a response-language value.
    Omitting ``lang`` uses Photon's default/local place names.
    """
    normalized_query = " ".join((query or "").strip().split())
    if not normalized_query:
        raise ValueError("query must not be empty")
    if limit <= 0:
        raise ValueError("limit must be positive")
    return {
        "q": normalized_query,
        "limit": int(limit),
        "countrycode": PHOTON_COUNTRY_CODE,
    }
