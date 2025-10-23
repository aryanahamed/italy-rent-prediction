"""
Utility functions for enhanced features in the rent prediction app.
Includes similar property search, historical trends, and affordability calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import streamlit as st


@st.cache_data(ttl=3600)
def load_rental_data():
    """
    Load and cache the rental dataset to avoid redundant file reads.
    Cache expires after 1 hour.
    
    Returns:
        DataFrame with rental data and standardized column names
    """
    data_path = Path(__file__).parent / 'rents_clean.csv' / 'rents_clean.csv'
    df = pd.read_csv(data_path)
    
    # Rename columns to standardized format
    df.columns = ['region', 'city', 'neighborhood', 'price', 'datetime', 'parking_spots', 
                  'bathrooms_per_room', 'bathrooms', 'rooms', 'top_floor', 'condition', 
                  'energy_class', 'sea_view', 'central_heating', 'area', 'furnished', 
                  'balcony', 'tv_system', 'external_exposure', 'fiber_optic', 'electric_gate', 
                  'cellar', 'shared_garden', 'private_garden', 'alarm_system', 'concierge', 
                  'pool', 'villa', 'entire_property', 'apartment', 'penthouse', 'loft', 'attic']
    
    return df


def find_similar_properties(
    city: str,
    rooms: int,
    area: float,
    price: float,
    top_n: int = 5
) -> List[Dict]:
    """
    Find similar properties from the dataset based on user criteria.
    
    Args:
        city: City name
        rooms: Number of rooms
        area: Area in square meters (in log space from prediction)
        price: Predicted price
        top_n: Number of similar properties to return
        
    Returns:
        List of similar property dictionaries
    """
    try:
        # Issue #2 fix: Use cached data loader
        df = load_rental_data()
        
        # Convert area from log space to normal space
        area_sqm = np.expm1(area)
        
        # Filter criteria
        df_filtered = df[
            (df['city'].str.lower() == city.lower()) &
            (df['rooms'] >= rooms - 1) & 
            (df['rooms'] <= rooms + 1) &
            (df['area'].notna()) &
            (df['area'] >= area_sqm * 0.8) & 
            (df['area'] <= area_sqm * 1.2) &
            (df['price'].notna()) &
            (df['price'] > 0)
        ].copy()
        
        # If city filter yields nothing, try without city constraint
        if len(df_filtered) < top_n:
            df_filtered = df[
                (df['rooms'] >= rooms - 1) & 
                (df['rooms'] <= rooms + 1) &
                (df['area'].notna()) &
                (df['area'] >= area_sqm * 0.8) & 
                (df['area'] <= area_sqm * 1.2) &
                (df['price'].notna()) &
                (df['price'] > 0)
            ].copy()
        
        if len(df_filtered) == 0:
            return []
        
        # Calculate similarity score based on area and rooms
        df_filtered['area_diff'] = abs(df_filtered['area'] - area_sqm)
        df_filtered['rooms_diff'] = abs(df_filtered['rooms'] - rooms)
        df_filtered['price_diff'] = abs(df_filtered['price'] - price)
        
        # Normalize and combine scores
        df_filtered['similarity_score'] = (
            (1 - df_filtered['area_diff'] / df_filtered['area_diff'].max()) * 0.4 +
            (1 - df_filtered['rooms_diff'] / max(df_filtered['rooms_diff'].max(), 1)) * 0.3 +
            (1 - df_filtered['price_diff'] / df_filtered['price_diff'].max()) * 0.3
        )
        
        # Sort by similarity and get top N
        df_similar = df_filtered.nlargest(top_n, 'similarity_score')
        
        # Convert to list of dictionaries
        similar_properties = []
        for _, row in df_similar.iterrows():
            similar_properties.append({
                'city': row['city'] if pd.notna(row['city']) else 'Unknown',
                'neighborhood': row['neighborhood'] if pd.notna(row['neighborhood']) else 'Unknown',
                'price': float(row['price']),
                'rooms': int(row['rooms']),
                'area': float(row['area']),
                'bathrooms': int(row['bathrooms']) if pd.notna(row['bathrooms']) else 0,
                'energy_class': row['energy_class'] if pd.notna(row['energy_class']) else 'N/A',
                'furnished': 'Yes' if row['furnished'] == 1 else 'No',
                'balcony': 'Yes' if row['balcony'] == 1 else 'No',
                'parking': 'Yes' if row['parking_spots'] == 1 else 'No',
                'condition': row['condition'] if pd.notna(row['condition']) else 'N/A',
                'similarity_score': float(row['similarity_score'])
            })
        
        return similar_properties
    
    except Exception as e:
        print(f"Error finding similar properties: {e}")
        return []


def get_historical_price_trends(
    city: str,
    neighborhood: Optional[str] = None,
    min_samples: int = 10
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Get historical price trends for a city or neighborhood.
    
    Args:
        city: City name
        neighborhood: Optional neighborhood name for more specific trends
        min_samples: Minimum number of data points required
        
    Returns:
        Tuple of (DataFrame with time series data, statistics dictionary)
    """
    try:
        # Issue #2 fix: Use cached data loader
        df = load_rental_data()
        
        # Filter by city and optionally neighborhood
        if neighborhood:
            df_filtered = df[
                (df['city'].str.lower() == city.lower()) &
                (df['neighborhood'].str.lower() == neighborhood.lower()) &
                (df['price'].notna()) &
                (df['datetime'].notna())
            ].copy()
        else:
            df_filtered = df[
                (df['city'].str.lower() == city.lower()) &
                (df['price'].notna()) &
                (df['datetime'].notna())
            ].copy()
        
        if len(df_filtered) < min_samples:
            return None, {}
        
        # Convert datetime column
        df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['datetime'])
        
        if len(df_filtered) < min_samples:
            return None, {}
        
        # Group by month and calculate average price
        df_filtered['year_month'] = df_filtered['datetime'].dt.to_period('M')
        monthly_avg = df_filtered.groupby('year_month').agg({
            'price': ['mean', 'median', 'count']
        }).reset_index()
        
        monthly_avg.columns = ['year_month', 'avg_price', 'median_price', 'count']
        monthly_avg['year_month'] = monthly_avg['year_month'].dt.to_timestamp()
        
        # Calculate statistics
        if len(monthly_avg) >= 2:
            first_price = monthly_avg.iloc[0]['avg_price']
            last_price = monthly_avg.iloc[-1]['avg_price']
            price_change = last_price - first_price
            price_change_pct = (price_change / first_price * 100) if first_price > 0 else 0
            
            # Calculate trend (simple linear regression)
            x = np.arange(len(monthly_avg))
            y = monthly_avg['avg_price'].values
            trend_coefficient = np.polyfit(x, y, 1)[0]  # Slope of the line
            
            stats = {
                'first_price': float(first_price),
                'last_price': float(last_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'trend': 'rising' if trend_coefficient > 0 else 'falling',
                'trend_coefficient': float(trend_coefficient),
                'data_points': len(monthly_avg),
                'total_listings': int(df_filtered['price'].count()),
                'avg_price_overall': float(df_filtered['price'].mean()),
                'min_price': float(df_filtered['price'].min()),
                'max_price': float(df_filtered['price'].max())
            }
        else:
            stats = {
                'avg_price_overall': float(df_filtered['price'].mean()),
                'min_price': float(df_filtered['price'].min()),
                'max_price': float(df_filtered['price'].max()),
                'total_listings': int(df_filtered['price'].count())
            }
        
        return monthly_avg, stats
    
    except Exception as e:
        print(f"Error getting historical trends: {e}")
        return None, {}


def calculate_affordability(
    rent_price: float,
    region: Optional[str] = None
) -> Dict:
    """
    Calculate rent affordability based on the 30% rule and Italian average salaries.
    
    Args:
        rent_price: Monthly rent price in euros
        region: Optional region for more specific salary data
        
    Returns:
        Dictionary with affordability metrics
    """
    # Italian average monthly net salaries by region (approximate 2023-2024 data)
    # Source: ISTAT (Italian National Institute of Statistics)
    regional_salaries = {
        'lombardia': 2000,
        'trentino-alto adige': 1950,
        'valle d\'aosta': 1900,
        'lazio': 1850,
        'emilia-romagna': 1850,
        'veneto': 1750,
        'friuli-venezia giulia': 1750,
        'piemonte': 1700,
        'liguria': 1700,
        'toscana': 1650,
        'marche': 1550,
        'umbria': 1500,
        'abruzzo': 1450,
        'sardegna': 1400,
        'campania': 1350,
        'puglia': 1300,
        'sicilia': 1300,
        'basilicata': 1300,
        'molise': 1250,
        'calabria': 1250
    }
    
    # National average
    national_avg_salary = 1650
    
    # Get region-specific or national average
    if region:
        avg_salary = regional_salaries.get(region.lower(), national_avg_salary)
    else:
        avg_salary = national_avg_salary
    
    # Calculate required income (30% rule)
    required_income = rent_price / 0.30
    
    # Calculate percentage of average salary
    pct_of_avg_salary = (rent_price / avg_salary * 100) if avg_salary > 0 else 0
    
    # Determine affordability level
    if pct_of_avg_salary <= 30:
        affordability_level = 'Highly Affordable'
        affordability_color = 'green'
        affordability_emoji = '🟢'
    elif pct_of_avg_salary <= 40:
        affordability_level = 'Affordable'
        affordability_color = 'lightgreen'
        affordability_emoji = '🟡'
    elif pct_of_avg_salary <= 50:
        affordability_level = 'Moderate'
        affordability_color = 'orange'
        affordability_emoji = '🟠'
    else:
        affordability_level = 'Challenging'
        affordability_color = 'red'
        affordability_emoji = '🔴'
    
    return {
        'rent_price': float(rent_price),
        'required_income': float(required_income),
        'avg_salary': float(avg_salary),
        'pct_of_avg_salary': float(pct_of_avg_salary),
        'affordability_level': affordability_level,
        'affordability_color': affordability_color,
        'affordability_emoji': affordability_emoji,
        'is_affordable': pct_of_avg_salary <= 30
    }


def generate_prediction_report(
    prediction_data: Dict,
    similar_properties: List[Dict],
    affordability: Dict,
    historical_stats: Optional[Dict] = None
) -> str:
    """
    Generate a formatted text report for download.
    
    Args:
        prediction_data: Prediction results from session state
        similar_properties: List of similar properties
        affordability: Affordability calculation results
        historical_stats: Optional historical price statistics
        
    Returns:
        Formatted text report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ITALY RENT PREDICTION - DETAILED REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Prediction Summary
    report_lines.append("-" * 70)
    report_lines.append("PREDICTION SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Location: {prediction_data.get('address', 'Unknown')}")
    report_lines.append(f"Predicted Rent: €{prediction_data['euro_est']:.2f} per month")
    report_lines.append(f"Confidence Score: {prediction_data['confidence_score']:.1f}%")
    report_lines.append(f"95% Confidence Interval: €{prediction_data['lower_bound']:.2f} - €{prediction_data['upper_bound']:.2f}")
    range_width = prediction_data['upper_bound'] - prediction_data['lower_bound']
    report_lines.append(f"Range Width: €{range_width:.2f} (±{(range_width / prediction_data['euro_est'] * 100 / 2):.1f}%)")
    report_lines.append("")
    
    # Feature Contributions
    report_lines.append("-" * 70)
    report_lines.append("TOP FEATURES DRIVING THIS PRICE")
    report_lines.append("-" * 70)
    for i, contrib in enumerate(prediction_data['top_contributors'], 1):
        impact = contrib['contribution_euro']
        direction = "increases" if impact > 0 else "decreases"
        report_lines.append(f"{i}. {contrib['feature']}: {direction} rent by €{abs(impact):.2f}")
    report_lines.append("")
    
    # Affordability Analysis
    report_lines.append("-" * 70)
    report_lines.append("AFFORDABILITY ANALYSIS")
    report_lines.append("-" * 70)
    report_lines.append(f"Affordability Level: {affordability['affordability_level']}")
    report_lines.append(f"Required Monthly Income (30% rule): €{affordability['required_income']:.2f}")
    report_lines.append(f"Average Italian Salary: €{affordability['avg_salary']:.2f}")
    report_lines.append(f"Rent as % of Average Salary: {affordability['pct_of_avg_salary']:.1f}%")
    if affordability['is_affordable']:
        report_lines.append("✓ This rent is within the affordable range (≤30% of income)")
    else:
        report_lines.append("⚠ This rent exceeds the recommended 30% affordability threshold")
    report_lines.append("")
    
    # Historical Trends (if available)
    if historical_stats and historical_stats:
        report_lines.append("-" * 70)
        report_lines.append("HISTORICAL PRICE TRENDS")
        report_lines.append("-" * 70)
        if 'trend' in historical_stats:
            report_lines.append(f"Market Trend: {historical_stats['trend'].upper()}")
            if 'price_change_pct' in historical_stats:
                report_lines.append(f"Price Change: {historical_stats['price_change_pct']:+.1f}% over time")
        if 'avg_price_overall' in historical_stats:
            report_lines.append(f"Average Price in Area: €{historical_stats['avg_price_overall']:.2f}")
        if 'total_listings' in historical_stats:
            report_lines.append(f"Total Listings Analyzed: {historical_stats['total_listings']}")
        report_lines.append("")
    
    # Similar Properties
    if similar_properties:
        report_lines.append("-" * 70)
        report_lines.append(f"SIMILAR PROPERTIES ({len(similar_properties)} found)")
        report_lines.append("-" * 70)
        for i, prop in enumerate(similar_properties, 1):
            report_lines.append(f"\n{i}. {prop['neighborhood']}, {prop['city']}")
            report_lines.append(f"   Price: €{prop['price']:.2f}/month")
            report_lines.append(f"   Rooms: {prop['rooms']} | Area: {prop['area']:.0f} m²")
            report_lines.append(f"   Bathrooms: {prop['bathrooms']} | Energy Class: {prop['energy_class']}")
            report_lines.append(f"   Furnished: {prop['furnished']} | Balcony: {prop['balcony']} | Parking: {prop['parking']}")
        report_lines.append("")
    
    # Disclaimer
    report_lines.append("-" * 70)
    report_lines.append("DISCLAIMER")
    report_lines.append("-" * 70)
    report_lines.append("This prediction is generated by a machine learning model trained on")
    report_lines.append("historical rental data from Italy. Actual rental prices may vary based")
    report_lines.append("on market conditions, property condition, and other factors not captured")
    report_lines.append("in the model. This report is for informational purposes only.")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("End of Report")
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines)
