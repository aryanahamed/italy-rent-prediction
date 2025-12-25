import pandas as pd
import numpy as np
import json

from utils import COLUMN_NAMES

# Load original data
df = pd.read_csv('rents_clean.csv/rents_clean.csv')
df.columns = COLUMN_NAMES

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
