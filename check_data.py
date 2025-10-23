import pandas as pd
import numpy as np
import json

# Load original data
df = pd.read_csv('rents_clean.csv/rents_clean.csv')
df.columns = ['region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots', 
              'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition', 
              'energy_class', 'sea_view', 'central_heating', 'area', 'furnished', 
              'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate', 
              'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge', 
              'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic']

print(f"Price sample (first 5): {df['price'].head().tolist()}")
print(f"Price range: {df['price'].min()} - {df['price'].max()}")
print(f"Total neighborhoods: {df['neighborhood'].nunique()}")
print(f"Non-null neighborhoods: {df['neighborhood'].notna().sum()}")
print(f"Sample neighborhoods: {df['neighborhood'].value_counts().head(10).to_dict()}")

# Load geocoding cache
with open('geocoding_cache/neighborhood_coordinates.json', 'r') as f:
    geo_data = json.load(f)

print(f"\nGeocoded neighborhoods: {len(geo_data)}")

# Check overlap
df_neighborhoods = set(df['neighborhood'].dropna().unique())
geo_neighborhoods = set(geo_data.keys())
overlap = df_neighborhoods.intersection(geo_neighborhoods)

print(f"Overlap between data and geocoding: {len(overlap)} neighborhoods")
print(f"Can create heatmap: {len(overlap) > 0}")
