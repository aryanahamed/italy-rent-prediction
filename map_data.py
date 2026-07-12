"""
Map data processing module for neighborhood price heatmap.
Handles loading and processing of neighborhood coordinates and rental prices.
"""
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from config import ITALY_BOUNDS, ITALY_CENTER_COORDS, MAX_CLUSTER_PROPERTIES, MIN_RENT
from utils import (
    COLUMN_NAMES,
    load_neighborhood_coordinates,
    load_rental_data,
)


def _format_binary_value(value) -> str:
    if pd.isna(value):
        return 'Unknown'
    if value == 1:
        return 'Yes'
    if value == 0:
        return 'No'
    return 'Unknown'

# Unpack bounds tuple for easy use
ITALY_LAT_MIN, ITALY_LAT_MAX, ITALY_LON_MIN, ITALY_LON_MAX = ITALY_BOUNDS

# Default map center (MU-10: replaced pointless cached function with constant)
# ITALY_CENTER_COORDS is imported from config


def _validate_coords(lat: float, lon: float) -> bool:
    """Check if coordinates fall within Italy bounds."""
    return (ITALY_LAT_MIN <= lat <= ITALY_LAT_MAX and
            ITALY_LON_MIN <= lon <= ITALY_LON_MAX)


@st.cache_data(ttl=3600)
def load_neighborhood_price_data():
    """
    Load and process legacy neighborhood data for heatmap visualization.

    The public app keeps this view disabled until coordinates are contextualized
    by city and region. This loader remains for migration and contract tests.

    Returns:
        list: List of [latitude, longitude, normalized_price] for heatmap
        dict: Statistics about the data (min, max, avg prices)
        list: Details about each valid neighborhood
    """
    # --- Load geocoding coordinates (MU-15: shared loader, MU-5: error handling) ---
    try:
        neighborhood_coords = load_neighborhood_coordinates()
    except FileNotFoundError:
        st.error(
            "Neighborhood coordinates file not found. "
            "Please ensure 'geocoding_cache/neighborhood_coordinates.json' exists."
        )
        return [], {}, []
    except Exception as e:
        st.error(f"Failed to load neighborhood coordinates: {e}")
        return [], {}, []

    # --- Load rental data (MU-14: shared loader, MU-5: error handling) ---
    try:
        df = load_rental_data()
    except FileNotFoundError:
        st.error(
            "Rental data file not found. "
            "Please ensure 'rents_clean.csv/rents_clean.csv' exists."
        )
        return [], {}, []
    except Exception as e:
        st.error(f"Failed to load rental data: {e}")
        return [], {}, []

    # MU-6: Drop rows with missing neighborhood before groupby
    df = df.dropna(subset=['neighborhood'])
    df = df[df['price'] >= MIN_RENT]

    # Calculate average price per neighborhood
    neighborhood_avg_prices = df.groupby('neighborhood')['price'].mean().to_dict()

    # Build heatmap data: [lat, lon, normalized_intensity]
    heatmap_data = []
    valid_neighborhoods = []

    # Find price range for normalization (MU-7)
    all_avg_prices = list(neighborhood_avg_prices.values())
    if all_avg_prices:
        min_price = min(all_avg_prices)
        max_price = max(all_avg_prices)
        price_range = max_price - min_price if max_price > min_price else 1.0
    else:
        min_price = 0.0
        max_price = 0.0
        price_range = 1.0

    for neighborhood, coords in neighborhood_coords.items():
        if neighborhood in neighborhood_avg_prices:
            lat, lon = coords
            avg_price = neighborhood_avg_prices[neighborhood]

            # Validate coordinates using shared bounds (MU-12)
            if _validate_coords(lat, lon):
                # Normalize average price to 0-1 range (MU-7)
                normalized = (avg_price - min_price) / price_range
                heatmap_data.append([lat, lon, normalized])
                valid_neighborhoods.append({
                    'neighborhood': neighborhood,
                    'latitude': lat,
                    'longitude': lon,
                    'avg_price': avg_price,
                    'normalized_price': normalized,
                    'property_count': len(df[df['neighborhood'] == neighborhood])
                })

    # Calculate statistics using actual euro prices from valid_neighborhoods
    # (BUG C1 fix: was using normalized 0-1 values from heatmap_data instead of euro amounts)
    all_prices = [n['avg_price'] for n in valid_neighborhoods]
    stats = {
        'min_price': min(all_prices) if all_prices else 0,
        'max_price': max(all_prices) if all_prices else 0,
        'avg_price': sum(all_prices) / len(all_prices) if all_prices else 0,
        'neighborhood_count': len(valid_neighborhoods),
    }

    return heatmap_data, stats, valid_neighborhoods


@st.cache_data(ttl=3600)
def load_property_cluster_data(
    price_min: float = 0,
    price_max: float = 10000,
    rooms_min: int = 1,
    rooms_max: int = 10,
    random_state: Optional[int] = 42,  # MU-11: parameterised, default 42
):
    """
    Load property data for cluster visualization with filtering options.

    Args:
        price_min: Minimum price filter
        price_max: Maximum price filter
        rooms_min: Minimum rooms filter
        rooms_max: Maximum rooms filter
        random_state: Seed for random sampling (None = no fixed seed)

    Returns:
        list: List of property dictionaries with location and details
    """
    # MU-15: Use shared JSON loader with error handling (MU-5)
    try:
        neighborhood_coords = load_neighborhood_coordinates()
    except FileNotFoundError:
        st.error(
            "Neighborhood coordinates file not found. "
            "Please ensure 'geocoding_cache/neighborhood_coordinates.json' exists."
        )
        return []
    except Exception as e:
        st.error(f"Failed to load neighborhood coordinates: {e}")
        return []

    # MU-14: Use shared CSV loader with error handling (MU-5)
    try:
        df = load_rental_data()
    except FileNotFoundError:
        st.error(
            "Rental data file not found. "
            "Please ensure 'rents_clean.csv/rents_clean.csv' exists."
        )
        return []
    except Exception as e:
        st.error(f"Failed to load rental data: {e}")
        return []

    # Clean and filter data — log how many rows dropped (MU-9)
    before_drop = len(df)
    df = df.dropna(subset=['price', 'neighborhood', 'rooms'])
    after_drop = len(df)
    dropped = before_drop - after_drop
    if dropped > 0:
        st.caption(f"⚠️ Dropped {dropped} rows with missing price/neighborhood/rooms data.")

    effective_price_min = max(price_min, MIN_RENT)
    df = df[(df['price'] >= effective_price_min) & (df['price'] <= price_max)]
    df = df[(df['rooms'] >= rooms_min) & (df['rooms'] <= rooms_max)]

    # Sample data to avoid overwhelming the map (max 2000 properties)
    max_props = MAX_CLUSTER_PROPERTIES
    if len(df) > max_props:
        df = df.sample(n=max_props, random_state=random_state)

    # Build property list
    properties = []
    for _, row in df.iterrows():
        neighborhood = row['neighborhood']

        # Get coordinates from geocoding cache
        if neighborhood in neighborhood_coords:
            lat, lon = neighborhood_coords[neighborhood]

            # Validate coordinates using shared bounds (MU-12)
            if _validate_coords(lat, lon):
                property_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'price': row['price'],
                    'rooms': int(row['rooms']) if pd.notna(row['rooms']) else 0,
                    'area': row['area'] if pd.notna(row['area']) else 'N/A',
                    'city': row['city'] if pd.notna(row['city']) else 'Unknown',
                    'neighborhood': neighborhood,
                    'energy_class': row['energy_class'] if pd.notna(row['energy_class']) else 'N/A',
                    'furnished': _format_binary_value(row['furnished']),
                    'bathrooms': int(row['bathrooms']) if pd.notna(row['bathrooms']) else 0,
                    'balcony': _format_binary_value(row['balcony']),
                    'parking': _format_binary_value(row['parking_spots']),
                }
                properties.append(property_data)

    return properties


def get_price_category(price: float) -> Dict[str, Any]:
    """
    Categorize price into ranges for color coding.

    Args:
        price: Monthly rent price in euros

    Returns:
        dict: Category info with color and label
    """
    if price < 0:
        raise ValueError("price must be non-negative")
    if price < 500:
        return {'color': 'green', 'label': 'Under €500', 'icon': 'home'}
    elif price < 1000:
        return {'color': 'blue', 'label': '€500–<€1,000', 'icon': 'home'}
    elif price < 1500:
        return {'color': 'orange', 'label': '€1,000–<€1,500', 'icon': 'home'}
    elif price < 2500:
        return {'color': 'red', 'label': '€1,500–<€2,500', 'icon': 'star'}
    else:
        return {'color': 'purple', 'label': '€2,500 or more', 'icon': 'star'}
