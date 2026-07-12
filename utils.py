"""
Shared utility functions for data loading and preprocessing.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

from config import DATA_DIR, MIN_RENT


# Column mapping for the rental dataset (Italian -> English)
COLUMN_NAMES = [
    'region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots',
    'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition',
    'energy_class', 'sea_view', 'central_heating', 'area', 'furnished',
    'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate',
    'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge',
    'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic'
]

SOURCE_COLUMN_NAMES = [
    'regione', 'citta', 'quartiere', 'prezzo', 'datetime', 'posti auto',
    'bagni per stanza', 'bagni', 'stanze', 'ultimo piano', 'stato',
    'classe energetica', 'vista mare', 'riscaldamento centralizzato',
    'superficie', 'arredato', 'balcone', 'impianto tv',
    'esposizione esterna', 'fibra ottica', 'cancello elettrico', 'cantina',
    'giardino comune', 'giardino privato', 'impianto allarme', 'portiere',
    'piscina', 'villa', 'intera proprieta', 'appartamento', 'attico', 'loft',
    'mansarda'
]


def remove_price_outliers(df: pd.DataFrame, column: str = 'price',
                          method: str = 'iqr', min_rent: float = MIN_RENT) -> pd.DataFrame:
    """
    Remove price outliers using IQR or z-score method.

    Both methods return consistently filtered data (same pattern).
    A minimum rent threshold prevents unrealistically cheap prices.

    Args:
        df: Input dataframe
        column: Column name to filter on
        method: 'iqr' (Interquartile Range) or 'zscore' (3-sigma threshold)
        min_rent: Minimum realistic rent in EUR (filters €1 errors, etc.)

    Returns:
        DataFrame with outliers removed; same shape as input minus filtered rows

    Raises:
        ValueError: If an unsupported method string is provided.
    """
    if df.empty:
        return df
    if column not in df.columns:
        raise ValueError(f"Required outlier column '{column}' is missing")

    # Remove NaN values first
    df_clean = df.dropna(subset=[column]).copy()

    if len(df_clean) == 0:
        return df_clean

    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        iqr = Q3 - Q1

        # Standard statistical outlier bounds
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr

        # Apply min_rent filter to avoid €1 ultra-cheap prices (MU-8)
        effective_lower = max(lower_bound, min_rent)

        df_filtered = df_clean[
            (df_clean[column] >= effective_lower) &
            (df_clean[column] <= upper_bound)
        ].copy()
        return df_filtered

    elif method == 'zscore':
        # Implement basic z-score outlier removal (MU-2)
        mean = df_clean[column].mean()
        std = df_clean[column].std()

        if not np.isfinite(std) or std == 0:
            return df_clean[df_clean[column] >= min_rent].copy()

        z_lower = mean - 3 * std
        z_upper = mean + 3 * std

        effective_lower = max(z_lower, min_rent)

        df_filtered = df_clean[
            (df_clean[column] >= effective_lower) &
            (df_clean[column] <= z_upper)
        ].copy()
        return df_filtered

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'iqr' or 'zscore'."
        )


# NOTE: TTL=3600 means data is cached for 1 hour before auto-refresh.
# If the underlying CSV changes more frequently, reduce this value (MU-4).
@st.cache_data(ttl=3600)
def load_rental_data() -> pd.DataFrame:
    """
    Load and cache the rental dataset.

    The CSV lives inside a directory named 'rents_clean.csv'
    (DATA_DIR / 'rents_clean.csv' / 'rents_clean.csv').

    Returns:
        DataFrame with renamed columns per COLUMN_NAMES.
    """
    # MU-1: Use shared DATA_DIR constant instead of bogus path construction
    csv_file = DATA_DIR / 'rents_clean.csv' / 'rents_clean.csv'
    df = pd.read_csv(csv_file)
    incoming = list(df.columns)
    if incoming == SOURCE_COLUMN_NAMES:
        df.columns = COLUMN_NAMES
    elif incoming != COLUMN_NAMES:
        raise ValueError(
            "Rental CSV schema mismatch. Expected the canonical Italian or English "
            f"33-column schema, received: {incoming}"
        )
    return df


# Shared cached loader for neighborhood coordinates (MU-15)
@st.cache_data(ttl=3600)
def load_neighborhood_coordinates() -> Dict[str, List[float]]:
    """
    Load and cache the neighborhood geocoding coordinates JSON.

    File location: DATA_DIR / 'geocoding_cache' / 'neighborhood_coordinates.json'

    Returns:
        Dictionary mapping neighborhood names to [latitude, longitude].
    """
    json_path = DATA_DIR / 'geocoding_cache' / 'neighborhood_coordinates.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        coords = json.load(f)
    return coords
