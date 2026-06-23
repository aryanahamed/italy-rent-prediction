"""
Utility functions for enhanced features in the rent prediction app.
Includes similar property search, historical trends, and affordability calculations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import streamlit as st

from utils import remove_price_outliers, load_rental_data


def find_similar_properties(
    city: str,
    rooms: int,
    area: float,
    price: float,
    top_n: int = 5
) -> List[Dict]:
    """
    Find similar properties from the dataset based on user criteria.
    Filters out extreme price outliers to ensure reliable comparisons.
    
    Args:
        city: City name
        rooms: Number of rooms
        area: Area in square meters
        price: Predicted price
        top_n: Number of similar properties to return
        
    Returns:
        List of similar property dictionaries
    """
    try:
        # Issue #2 fix: Use cached data loader
        df = load_rental_data()
        
        # Remove price outliers to ensure reliable comparisons
        df = remove_price_outliers(df, column='price', method='iqr')
        
        # Filter criteria
        df_filtered = df[
            (df['city'].str.lower() == city.lower()) &
            (df['rooms'] >= rooms - 1) & 
            (df['rooms'] <= rooms + 1) &
            (df['area'].notna()) &
            (df['area'] >= area * 0.8) & 
            (df['area'] <= area * 1.2) &
            (df['price'].notna()) &
            (df['price'] > 0)
        ].copy()
        
        # If city filter yields nothing, try without city constraint
        city_matched = True
        if len(df_filtered) < top_n:
            df_filtered = df[
                (df['rooms'] >= rooms - 1) & 
                (df['rooms'] <= rooms + 1) &
                (df['area'].notna()) &
                (df['area'] >= area * 0.8) & 
                (df['area'] <= area * 1.2) &
                (df['price'].notna()) &
                (df['price'] > 0)
            ].copy()
            city_matched = False
        
        if len(df_filtered) == 0:
            return []
        
        # Calculate similarity score based on area and rooms
        df_filtered['area_diff'] = abs(df_filtered['area'] - area)
        df_filtered['rooms_diff'] = abs(df_filtered['rooms'] - rooms)
        df_filtered['price_diff'] = abs(df_filtered['price'] - price)
        
        # Normalize and combine scores
        df_filtered['similarity_score'] = (
            (1 - df_filtered['area_diff'] / max(df_filtered['area_diff'].max(), 1)) * 0.4 +
            (1 - df_filtered['rooms_diff'] / max(df_filtered['rooms_diff'].max(), 1)) * 0.3 +
            (1 - df_filtered['price_diff'] / max(df_filtered['price_diff'].max(), 1)) * 0.3
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
                'similarity_score': float(row['similarity_score']),
                'city_matched': city_matched
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
        
        # Remove price outliers for cleaner trend analysis
        df = remove_price_outliers(df, column='price', method='iqr')
        
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
        n_before = len(df_filtered)
        df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'], errors='coerce')
        df_filtered = df_filtered.dropna(subset=['datetime'])
        n_dropped = n_before - len(df_filtered)
        if n_dropped > 0:
            print(f"Warning: Dropped {n_dropped} rows with invalid dates in historical trend data")
        
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
        # Require at least 4 data points for trend direction (2 points always give a perfect fit)
        if len(monthly_avg) >= 4:
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
    # URL: https://www.istat.it/en/income-and-productivity/
    # Last updated: 2024-06 (based on 2023 tax returns)
    # NOTE: These are static estimates. For production use, consider integrating
    # a live data source or updating annually.
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
    report_lines.append(f"Predicted Rent: €{prediction_data.get('euro_est', 0):.2f} per month")
    report_lines.append(f"Confidence Score: {prediction_data.get('confidence_score', 0):.1f}%")
    lower = prediction_data.get('lower_bound', 0)
    upper = prediction_data.get('upper_bound', 0)
    report_lines.append(f"95% Confidence Interval: €{lower:.2f} - €{upper:.2f}")
    range_width = upper - lower
    euro_est = prediction_data.get('euro_est', 1)
    pct = (range_width / euro_est * 100 / 2) if euro_est != 0 else 0
    report_lines.append(f"Range Width: €{range_width:.2f} (±{pct:.1f}%)")
    report_lines.append("")
    
    # Feature Contributions
    report_lines.append("-" * 70)
    report_lines.append("TOP FEATURES DRIVING THIS PRICE")
    report_lines.append("-" * 70)
    for i, contrib in enumerate(prediction_data.get('top_contributors', []), 1):
        impact = contrib.get('contribution_euro', 0)
        direction = "increases" if impact > 0 else "decreases"
        report_lines.append(f"{i}. {contrib.get('feature', 'Unknown')}: {direction} rent by €{abs(impact):.2f}")
    report_lines.append("")
    
    # Affordability Analysis
    report_lines.append("-" * 70)
    report_lines.append("AFFORDABILITY ANALYSIS")
    report_lines.append("-" * 70)
    report_lines.append(f"Affordability Level: {affordability.get('affordability_level', 'Unknown')}")
    report_lines.append(f"Required Monthly Income (30% rule): €{affordability.get('required_income', 0):.2f}")
    report_lines.append(f"Average Italian Salary: €{affordability.get('avg_salary', 0):.2f}")
    report_lines.append(f"Rent as % of Average Salary: {affordability.get('pct_of_avg_salary', 0):.1f}%")
    if affordability.get('is_affordable', False):
        report_lines.append("✓ This rent is within the affordable range (≤30% of income)")
    else:
        report_lines.append("⚠ This rent exceeds the recommended 30% affordability threshold")
    report_lines.append("")
    
    # Historical Trends (if available)
    if historical_stats:
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
