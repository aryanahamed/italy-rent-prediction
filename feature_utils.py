"""
User-facing comparable search, affordability references, and report helpers.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from config import MIN_RENT
from utils import load_rental_data


def _format_binary_value(value) -> str:
    """Preserve unknown values instead of silently presenting them as “No”."""
    if pd.isna(value):
        return 'Unknown'
    if value == 1:
        return 'Yes'
    if value == 0:
        return 'No'
    return 'Unknown'


def find_similar_properties(
    city: str,
    rooms: int,
    area: float,
    price: Optional[float] = None,
    top_n: int = 5
) -> List[Dict]:
    """
    Find archived same-city records with similar rooms and area.
    
    Args:
        city: City name
        rooms: Number of rooms
        area: Area in square meters
        price: Deprecated compatibility argument. It is deliberately ignored;
            the model prediction must never influence comparable selection.
        top_n: Number of similar properties to return
        
    Returns:
        List of similar property dictionaries
    """
    try:
        # Issue #2 fix: Use cached data loader
        df = load_rental_data()
        
        city = (city or '').strip()
        if not city or rooms <= 0 or area <= 0 or top_n <= 0:
            return []

        # Exact duplicate snapshots must not occupy multiple comparable slots.
        # A stable listing ID is still needed for cross-date deduplication.
        df = df.drop_duplicates().copy()
        
        # Filter criteria
        df_filtered = df[
            (df['city'].fillna('').str.casefold() == city.casefold()) &
            (df['rooms'] >= rooms - 1) & 
            (df['rooms'] <= rooms + 1) &
            (df['area'].notna()) &
            (df['area'] >= area * 0.8) & 
            (df['area'] <= area * 1.2) &
            (df['price'].notna()) &
            (df['price'] >= MIN_RENT)
        ].copy()
        
        if len(df_filtered) == 0:
            return []
        
        # Calculate similarity score based on area and rooms
        df_filtered['area_diff'] = abs(df_filtered['area'] - area)
        df_filtered['rooms_diff'] = abs(df_filtered['rooms'] - rooms)
        # Use fixed, interpretable distances. Normalizing by the candidate set
        # made scores change when unrelated rows were added or removed.
        area_relative_diff = df_filtered['area_diff'] / max(float(area), 1.0)
        rooms_relative_diff = df_filtered['rooms_diff'] / max(float(rooms), 1.0)
        df_filtered['similarity_score'] = (
            (1 - area_relative_diff.clip(upper=1.0)) * 0.7 +
            (1 - rooms_relative_diff.clip(upper=1.0)) * 0.3
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
                'furnished': _format_binary_value(row['furnished']),
                'balcony': _format_binary_value(row['balcony']),
                'parking': _format_binary_value(row['parking_spots']),
                'condition': row['condition'] if pd.notna(row['condition']) else 'N/A',
                'similarity_score': float(row['similarity_score']),
                'city_matched': True
            })
        
        return similar_properties
    
    except Exception as e:
        raise RuntimeError(f"Unable to calculate comparable properties: {e}") from e


def get_historical_price_trends(
    city: str,
    neighborhood: Optional[str] = None,
    min_samples: int = 10
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Refuse to manufacture a time series from the listing snapshot.

    Kept as a compatibility boundary for callers of older versions. Genuine
    trends require repeated observations with documented collection dates.
    """
    raise RuntimeError(
        "Historical trends are unavailable: the source is a listing snapshot, "
        "not a longitudinal dataset."
    )


def calculate_affordability(
    rent_price: float,
    monthly_household_income: Optional[float] = None,
) -> Dict:
    """
    Calculate a transparent rent-to-income reference.

    This is not a personal affordability verdict: it excludes utilities, taxes,
    allowances, debt, household composition, and other essential spending.
    
    Args:
        rent_price: Monthly rent price in euros
        monthly_household_income: Optional disposable household income in euros.
        
    Returns:
        Dictionary with affordability metrics
    """
    rent_price = float(rent_price)
    if not np.isfinite(rent_price) or rent_price < 0:
        raise ValueError("rent_price must be a finite, non-negative monthly amount")

    reference_pct = 30.0
    required_income = rent_price / (reference_pct / 100)
    income = None
    if monthly_household_income is not None:
        candidate = float(monthly_household_income)
        if not np.isfinite(candidate) or candidate <= 0:
            raise ValueError("monthly_household_income must be positive when provided")
        income = candidate

    pct_of_income = (rent_price / income * 100) if income is not None else None
    if pct_of_income is None:
        ratio_level = "Income not provided"
        ratio_band = "unknown"
        is_within_reference = None
    elif pct_of_income <= 30:
        ratio_level = "At or below 30% reference"
        ratio_band = "at_or_below_reference"
        is_within_reference = True
    elif pct_of_income <= 40:
        ratio_level = "Above 30% reference"
        ratio_band = "above_reference"
        is_within_reference = False
    else:
        ratio_level = "High rent-to-income ratio"
        ratio_band = "high_ratio"
        is_within_reference = False

    return {
        'rent_price': rent_price,
        'required_income': float(required_income),
        'monthly_household_income': income,
        'pct_of_income': float(pct_of_income) if pct_of_income is not None else None,
        'ratio_level': ratio_level,
        'ratio_band': ratio_band,
        'reference_pct': reference_pct,
        'is_within_reference': is_within_reference,
        # Compatibility keys for callers migrating from the old salary verdict.
        'avg_salary': income,
        'pct_of_avg_salary': float(pct_of_income) if pct_of_income is not None else None,
        'affordability_level': ratio_level,
        'is_affordable': is_within_reference,
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
    report_lines.append(f"Data through: {prediction_data.get('data_as_of', 'Unknown')}")
    report_lines.append(
        f"Input Stability Score: {prediction_data.get('stability_score', prediction_data.get('confidence_score', 0)):.1f}/100"
    )
    lower = prediction_data.get('lower_bound', 0)
    upper = prediction_data.get('upper_bound', 0)
    report_lines.append(f"Input Perturbation Range: €{lower:.2f} - €{upper:.2f}")
    range_width = upper - lower
    euro_est = prediction_data.get('euro_est', 1)
    pct = (range_width / euro_est * 100 / 2) if euro_est != 0 else 0
    report_lines.append(f"Range Width: €{range_width:.2f} (±{pct:.1f}%)")
    report_lines.append("")
    
    report_lines.append(
        "This range measures sensitivity to artificial input changes; it is not a prediction "
        "interval, probability, or accuracy guarantee."
    )
    report_lines.append("")

    # Submitted property snapshot
    submitted_inputs = prediction_data.get('inputs', {})
    if submitted_inputs:
        report_lines.append("-" * 70)
        report_lines.append("SUBMITTED PROPERTY")
        report_lines.append("-" * 70)
        for key, value in submitted_inputs.items():
            if key != 'monthly_household_income':
                report_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        report_lines.append("")

    # One-feature sensitivity checks
    report_lines.append("-" * 70)
    report_lines.append("ONE-FEATURE MODEL SENSITIVITY CHECKS")
    report_lines.append("-" * 70)
    for i, contrib in enumerate(prediction_data.get('top_contributors', []), 1):
        impact = contrib.get('contribution_euro', 0)
        direction = "higher" if impact > 0 else "lower" if impact < 0 else "unchanged"
        report_lines.append(
            f"{i}. {contrib.get('feature', 'Unknown')}: estimate is {direction} by €{abs(impact):.2f} "
            "versus the stated synthetic reference"
        )
    report_lines.append("These are one-at-a-time model sensitivity checks, not causal effects.")
    report_lines.append("")
    
    # Affordability Analysis
    report_lines.append("-" * 70)
    report_lines.append("AFFORDABILITY ANALYSIS")
    report_lines.append("-" * 70)
    report_lines.append(
        f"Monthly household income needed at the 30% reference: €{affordability.get('required_income', 0):.2f}"
    )
    pct_of_income = affordability.get('pct_of_income')
    if pct_of_income is not None:
        report_lines.append(
            f"Rent as % of submitted household income: {pct_of_income:.1f}%"
        )
        report_lines.append(f"Reference band: {affordability.get('ratio_level', 'Unknown')}")
    else:
        report_lines.append("Household income was not provided; no affordability verdict is made.")
    report_lines.append("This ratio excludes utilities and other household costs.")
    report_lines.append("")
    
    # The source is a listing snapshot, not a longitudinal market panel. Do not
    # emit a historical-trend claim even if a legacy caller passes statistics.
    
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
    report_lines.append("This estimate is generated by a machine learning model trained on an")
    report_lines.append("archived snapshot of advertised Italian rents. It is not a live quote.")
    report_lines.append("Actual rents may vary because of market changes, property details, and")
    report_lines.append("other factors the model does not capture. For information only.")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("End of Report")
    report_lines.append("=" * 70)
    
    return "\n".join(report_lines)
