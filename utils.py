"""
Shared utility functions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import streamlit as st


# Column mapping for the rental dataset (Italian → English)
COLUMN_NAMES = [
    'region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots',
    'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition',
    'energy_class', 'sea_view', 'central_heating', 'area', 'furnished',
    'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate',
    'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge',
    'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic'
]


def remove_price_outliers(df: pd.DataFrame, column: str = 'price', method: str = 'iqr') -> pd.DataFrame:
    """
    Remove price outliers using the IQR.
    """
    if df.empty or column not in df.columns:
        return df
    
    # Remove NaN values first
    df_clean = df.dropna(subset=[column]).copy()
    
    if len(df_clean) == 0:
        return df
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (standard statistical method)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep only values within bounds
        df_filtered = df_clean[(df_clean[column] >= lower_bound) & 
                              (df_clean[column] <= upper_bound)].copy()
        
        return df_filtered
    
    return df_clean


@st.cache_data(ttl=3600)
def load_rental_data() -> pd.DataFrame:
    """
    Load and cache the rental dataset
    """
    data_path = Path(__file__).parent / 'rents_clean.csv' / 'rents_clean.csv'
    df = pd.read_csv(data_path)
    df.columns = COLUMN_NAMES
    return df
