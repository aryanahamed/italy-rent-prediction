"""
Map data processing module for neighborhood price heatmap.
Handles loading and processing of neighborhood coordinates and rental prices.
"""

import pandas as pd
import json
import streamlit as st
from pathlib import Path


@st.cache_data(ttl=3600)
def load_neighborhood_price_data():
    """
    Load and process neighborhood data for heatmap visualization.
    
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
