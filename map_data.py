"""
Map data processing module for neighborhood price heatmap.
Handles loading and processing of neighborhood coordinates and rental prices.
"""

import pandas as pd
import json
import streamlit as st
from pathlib import Path


def remove_price_outliers(df: pd.DataFrame, column: str = 'price', method: str = 'iqr') -> pd.DataFrame:
    """
    Remove price outliers using the IQR (Interquartile Range) method.
    
    The IQR method removes values outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR],
    which is a standard statistical approach that removes extreme outliers while
    preserving legitimate market variations.
    
    Args:
        df: DataFrame with price data
        column: Column name to filter (default: 'price')
        method: Filtering method ('iqr' for Interquartile Range)
        
    Returns:
        DataFrame with outliers removed
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
def load_neighborhood_price_data():
    """
    Load and process neighborhood data for heatmap visualization.
    Filters price outliers to ensure representative neighborhood averages.
    
    Returns:
        list: List of [latitude, longitude, normalized_price] for heatmap
        dict: Statistics about the data (min, max, avg prices)
    """
    # Load geocoding coordinates
    geo_path = Path(__file__).parent / 'geocoding_cache' / 'neighborhood_coordinates.json'
    with open(geo_path, 'r', encoding='utf-8') as f:
        neighborhood_coords = json.load(f)
    
    # Load original rental data with neighborhood column
    data_path = Path(__file__).parent / 'rents_clean.csv' / 'rents_clean.csv'
    df = pd.read_csv(data_path)
    
    # Rename columns (Italian to English)
    df.columns = ['region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots', 
                  'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition', 
                  'energy_class', 'sea_view', 'central_heating', 'area', 'furnished', 
                  'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate', 
                  'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge', 
                  'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic']
    
    # Remove price outliers for accurate neighborhood averages
    df = remove_price_outliers(df, column='price', method='iqr')
    
    # Calculate average price per neighborhood
    neighborhood_avg_prices = df.groupby('neighborhood')['price'].mean().to_dict()
    
    # Build heatmap data: [lat, lon, intensity]
    heatmap_data = []
    valid_neighborhoods = []
    
    for neighborhood, coords in neighborhood_coords.items():
        if neighborhood in neighborhood_avg_prices:
            lat, lon = coords
            avg_price = neighborhood_avg_prices[neighborhood]
            
            # Only include valid coordinates (Italy is roughly 35-47°N, 6-19°E)
            if 35 <= lat <= 47 and 6 <= lon <= 19:
                heatmap_data.append([lat, lon, avg_price])
                valid_neighborhoods.append({
                    'neighborhood': neighborhood,
                    'latitude': lat,
                    'longitude': lon,
                    'avg_price': avg_price,
                    'property_count': len(df[df['neighborhood'] == neighborhood])
                })
    
    # Calculate statistics
    all_prices = [point[2] for point in heatmap_data]
    stats = {
        'min_price': min(all_prices) if all_prices else 0,
        'max_price': max(all_prices) if all_prices else 0,
        'avg_price': sum(all_prices) / len(all_prices) if all_prices else 0,
        'neighborhood_count': len(heatmap_data)
    }
    
    return heatmap_data, stats, valid_neighborhoods


@st.cache_data(ttl=3600)
def get_italy_center_coords():
    """
    Get center coordinates for Italy map.
    
    Returns:
        tuple: (latitude, longitude) for map center
    """
    # Center of Italy (approximately Rome area)
    return (41.9028, 12.4964)


@st.cache_data(ttl=3600)
def load_property_cluster_data(price_min=0, price_max=10000, rooms_min=1, rooms_max=10):
    """
    Load property data for cluster visualization with filtering options.
    
    Args:
        price_min: Minimum price filter
        price_max: Maximum price filter
        rooms_min: Minimum rooms filter
        rooms_max: Maximum rooms filter
    
    Returns:
        list: List of property dictionaries with location and details
    """
    # Load geocoding coordinates
    geo_path = Path(__file__).parent / 'geocoding_cache' / 'neighborhood_coordinates.json'
    with open(geo_path, 'r', encoding='utf-8') as f:
        neighborhood_coords = json.load(f)
    
    # Load rental data
    data_path = Path(__file__).parent / 'rents_clean.csv' / 'rents_clean.csv'
    df = pd.read_csv(data_path)
    
    # Rename columns
    df.columns = ['region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots', 
                  'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition', 
                  'energy_class', 'sea_view', 'central_heating', 'area', 'furnished', 
                  'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate', 
                  'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge', 
                  'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic']
    
    # Remove price outliers for accurate cluster representation
    df = remove_price_outliers(df, column='price', method='iqr')
    
    # Clean and filter data
    df = df.dropna(subset=['price', 'neighborhood', 'rooms'])
    df = df[(df['price'] >= price_min) & (df['price'] <= price_max)]
    df = df[(df['rooms'] >= rooms_min) & (df['rooms'] <= rooms_max)]
    
    # Sample data to avoid overwhelming the map (max 2000 properties)
    if len(df) > 2000:
        df = df.sample(n=2000, random_state=42)
    
    # Build property list
    properties = []
    for _, row in df.iterrows():
        neighborhood = row['neighborhood']
        
        # Get coordinates from geocoding cache
        if neighborhood in neighborhood_coords:
            lat, lon = neighborhood_coords[neighborhood]
            
            # Validate coordinates (Italy bounds)
            if 35 <= lat <= 47 and 6 <= lon <= 19:
                property_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'price': row['price'],
                    'rooms': int(row['rooms']) if pd.notna(row['rooms']) else 0,
                    'area': row['area'] if pd.notna(row['area']) else 'N/A',
                    'city': row['city'] if pd.notna(row['city']) else 'Unknown',
                    'neighborhood': neighborhood,
                    'energy_class': row['energy_class'] if pd.notna(row['energy_class']) else 'N/A',
                    'furnished': 'Yes' if row['furnished'] == 1 else 'No',
                    'bathrooms': int(row['bathrooms']) if pd.notna(row['bathrooms']) else 0,
                    'balcony': 'Yes' if row['balcony'] == 1 else 'No',
                    'parking': 'Yes' if row['parking_spots'] == 1 else 'No'
                }
                properties.append(property_data)
    
    return properties


def get_price_category(price):
    """
    Categorize price into ranges for color coding.
    
    Args:
        price: Monthly rent price in euros
        
    Returns:
        dict: Category info with color and label
    """
    if price < 500:
        return {'color': 'green', 'label': 'Budget (< €500)', 'icon': 'home'}
    elif price < 1000:
        return {'color': 'blue', 'label': 'Affordable (€500-€1000)', 'icon': 'home'}
    elif price < 1500:
        return {'color': 'orange', 'label': 'Mid-range (€1000-€1500)', 'icon': 'home'}
    elif price < 2500:
        return {'color': 'red', 'label': 'Premium (€1500-€2500)', 'icon': 'star'}
    else:
        return {'color': 'purple', 'label': 'Luxury (€2500+)', 'icon': 'star'}
