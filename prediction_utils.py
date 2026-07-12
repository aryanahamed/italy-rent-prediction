"""Prediction diagnostics and model sensitivity explanations."""
import numpy as np
from typing import Tuple, List, Dict
import warnings

from config import (
    MAX_AREA_M2,
    MAX_BATHROOMS,
    MAX_ROOMS,
    MIN_AREA_M2,
    MIN_BATHROOMS,
    MIN_ROOMS,
)
from model_contract import calculate_baths_per_room, calculate_rooms_per_area

warnings.filterwarnings('ignore')


# Binary feature names (from model.feature_names_in_) — used to identify
# features that are truly 0/1 flags and should not be perturbed during
# stability estimation, or assigned a zero (not present) reference during
# analysis. Value-based detection (features[0, idx] in [0, 1])
# misclassifies count features (bathrooms, rooms) when they happen to be 1.
# Both space-separated and underscore-separated variants are included
# since feature names may vary across model serialization.
BINARY_FEATURE_NAMES = frozenset({
    'parking spots', 'parking_spots', 'top floor', 'top_floor',
    'sea view', 'sea_view', 'central heating', 'central_heating',
    'furnished', 'balcony',
    'external exposure', 'external_exposure',
    'fiber optic', 'fiber_optic',
    'electric gate', 'electric_gate',
    'cellar', 'shared garden', 'shared_garden', 'pool',
    'tv_system', 'tv system',
    'private_garden', 'private garden',
    'alarm_system', 'alarm system',
    'concierge', 'villa', 'entire_property', 'entire property',
    'apartment', 'penthouse', 'loft', 'attic',
    'energy class_B', 'energy class_C', 'energy class_D',
    'energy class_E', 'energy class_F', 'energy class_G',
    'energy class_Unknown',
    'condition_buono / abitabile', 'condition_da ristrutturare',
    'condition_nuovo / in costruzione', 'condition_ottimo / ristrutturato'
})


class PredictionAnalyzer:
    """Analyze local input stability and one-feature model sensitivity."""
    
    def __init__(self, model, feature_medians=None):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            model: Trained XGBRegressor or RandomForestRegressor model
            feature_medians: Optional array of training feature medians for reference
                            computation. Continuous features are omitted when these
                            medians are unavailable.
        """
        self.model = model
        self.feature_names = list(getattr(
            model, 'feature_names_in_',
            [f'feature_{i}' for i in range(model.n_features_in_)]
        ))
        self.feature_medians = feature_medians
        self.feature_importances = model.feature_importances_
        
        # Detect model type
        self.model_type = type(model).__name__
        self.is_xgboost = 'XGB' in self.model_type
        self.is_random_forest = 'RandomForest' in self.model_type
        
    def calculate_confidence_score(self, features: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate an input-perturbation stability score.
        
        Area is varied by ±5%, and rooms/bathrooms by one within supported
        bounds. Derived ratios are recomputed so the model never receives an
        internally contradictory synthetic property.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound, stability_score).
            The range is an empirical perturbation range, not a calibrated
            prediction interval or probability statement.
        """
        # Ensure single-row input
        features = np.atleast_2d(features)
        if features.shape[0] > 1:
            features = features[:1]  # Use only first row if multiple provided
        
        # Get base prediction
        mean_prediction = self.model.predict(features)[0]
        
        name_to_index = {str(name): idx for idx, name in enumerate(self.feature_names)}
        scenarios = []

        def add_scenario(feature_name, value):
            idx = name_to_index.get(feature_name)
            if idx is None or value == features[0, idx]:
                return
            scenario = features.copy()
            scenario[0, idx] = value
            rooms_idx = name_to_index.get('rooms')
            bathrooms_idx = name_to_index.get('bathrooms')
            area_idx = name_to_index.get('area')
            rooms_area_idx = name_to_index.get('rooms_per_area')
            baths_room_idx = name_to_index.get('baths_per_room')
            if rooms_area_idx is not None and rooms_idx is not None and area_idx is not None:
                scenario[0, rooms_area_idx] = calculate_rooms_per_area(
                    scenario[0, rooms_idx], scenario[0, area_idx]
                )
            if baths_room_idx is not None and bathrooms_idx is not None and rooms_idx is not None:
                scenario[0, baths_room_idx] = calculate_baths_per_room(
                    scenario[0, bathrooms_idx], scenario[0, rooms_idx]
                )
            scenarios.append(scenario)

        if 'area' in name_to_index:
            area = float(features[0, name_to_index['area']])
            add_scenario('area', max(MIN_AREA_M2, area * 0.95))
            add_scenario('area', min(MAX_AREA_M2, area * 1.05))
        if 'rooms' in name_to_index:
            rooms = int(round(features[0, name_to_index['rooms']]))
            add_scenario('rooms', max(MIN_ROOMS, rooms - 1))
            add_scenario('rooms', min(MAX_ROOMS, rooms + 1))
        if 'bathrooms' in name_to_index:
            bathrooms = int(round(features[0, name_to_index['bathrooms']]))
            add_scenario('bathrooms', max(MIN_BATHROOMS, bathrooms - 1))
            add_scenario('bathrooms', min(MAX_BATHROOMS, bathrooms + 1))

        scenario_predictions = [self.model.predict(scenario)[0] for scenario in scenarios]
        lower_bound = min(scenario_predictions + [mean_prediction])
        upper_bound = max(scenario_predictions + [mean_prediction])

        lower_bound = min(float(lower_bound), float(mean_prediction))
        upper_bound = max(float(upper_bound), float(mean_prediction))

        # Compare width in euros, not log-target units.
        euro_prediction = np.expm1(mean_prediction)
        if euro_prediction != 0:
            interval_width = np.expm1(upper_bound) - np.expm1(lower_bound)
            relative_interval = interval_width / max(abs(euro_prediction), 1e-10)
            stability_score = np.clip(100 * (1 - relative_interval), 0, 100)
            stability_score = np.nan_to_num(stability_score, nan=0.0)
        else:
            stability_score = 0.0
        
        # Convert to Python native types for Streamlit compatibility
        return float(mean_prediction), float(lower_bound), float(upper_bound), float(stability_score)
    
    def get_feature_contributions(self, features: np.ndarray, base_prediction: float) -> List[Dict]:
        """
        Calculate individual feature contributions to the prediction.
        
        Uses a perturbation approach: measures how prediction changes when each
        feature is set to a stated reference value.
        
        Args:
            features: Input features for prediction
            base_prediction: The model's prediction for the input features (log scale)
            
        Returns:
            List of dictionaries with feature name, contribution, and impact
        """
        contributions = []
        
        # Ensure single-row input
        features = np.atleast_2d(features)
        if features.shape[0] > 1:
            features = features[:1]  # Use only first row if multiple provided
        
        features_flat = features.flatten()
        
        # Only use references with a defensible meaning. Binary features compare
        # against "not present" (0); continuous features require a supplied
        # training median and are omitted otherwise.
        baseline_values = np.full(len(self.feature_names), np.nan)
        for idx, _ in enumerate(features_flat):
            if self.feature_names[idx] in BINARY_FEATURE_NAMES:
                baseline_values[idx] = 0.0
            elif self.feature_medians is not None and idx < len(self.feature_medians):
                baseline_values[idx] = self.feature_medians[idx]
        
        # Calculate contribution for each feature
        for idx, feature_name in enumerate(self.feature_names):
            if np.isnan(baseline_values[idx]):
                continue
            # Skip if very low importance
            if self.feature_importances[idx] < 0.001:
                continue
            
            # Create perturbed features with this feature set to baseline
            features_perturbed = features.copy()
            features_perturbed[0, idx] = baseline_values[idx]
            
            # Get prediction without this feature (at baseline)
            pred_without = self.model.predict(features_perturbed)[0]
            
            # Contribution is the difference
            contribution_log = base_prediction - pred_without
            
            # Convert to euro
            euro_base = np.expm1(base_prediction)
            euro_without = np.expm1(pred_without)
            euro_contribution = euro_base - euro_without
            
            contributions.append({
                'feature': self._format_feature_name(feature_name),
                'raw_name': feature_name,
                'value': features_flat[idx],
                'baseline_value': baseline_values[idx],
                'importance': self.feature_importances[idx],
                'contribution_euro': euro_contribution,
                'contribution_log': contribution_log
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution_euro']), reverse=True)
        
        return contributions
    
    def _format_feature_name(self, feature_name: str) -> str:
        """Format feature name for display."""
        name_map = {
            'parking spots': 'Parking Spots',
            'bathrooms': 'Bathrooms',
            'rooms': 'Rooms',
            'energy class': 'Energy Class',
            'central heating': 'Central Heating',
            'area': 'Area',
            'furnished': 'Furnished',
            'balcony': 'Balcony',
            'external exposure': 'External Exposure',
            'fiber optic': 'Fiber Optic',
            'electric gate': 'Electric Gate',
            'shared garden': 'Shared Garden',
            'Building Layout': 'Building Layout',
            'Furnished and Central Heating': 'Furnished + Central Heating',
            'latitude': 'Location (Latitude)',
            'longitude': 'Location (Longitude)',
            'latitude_city': 'City Location (Lat)',
            'longitude_city': 'City Location (Lon)',
            'latitude_neighborhood': 'Neighborhood Location (Lat)',
            'longitude_neighborhood': 'Neighborhood Location (Lon)',
            'condition_ottimo / ristrutturato': 'Excellent Condition',
            'condition_buono / abitabile': 'Good Condition',
            'condition_da ristrutturare': 'To Be Renovated',
            'condition_nuovo / in costruzione': 'New/Under Construction',
            'rooms_per_area': 'Rooms per Area (m²)',
            'baths_per_room': 'Baths per Room',
            'amenity_score': 'Amenity Score',
            'top floor': 'Top Floor',
            'sea view': 'Sea View',
            'cellar': 'Cellar',
            'pool': 'Pool'
        }
        return name_map.get(feature_name, feature_name.replace('_', ' ').title())
    
    def get_top_contributors(self, contributions: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Get top N features contributing to the prediction.
        
        Args:
            contributions: List of feature contributions
            top_n: Number of top features to return
            
        Returns:
            List of top contributing features
        """
        # Filter out location features for cleaner display (combine their impact)
        # Use prefix-based matching to avoid hardcoded feature name lists
        non_location = [c for c in contributions
                        if abs(c['contribution_euro']) >= 1.0 and
                        not (c['raw_name'].startswith('latitude') or
                             c['raw_name'].startswith('longitude'))]
        location = [c for c in contributions 
                    if c['raw_name'].startswith('latitude') or 
                       c['raw_name'].startswith('longitude')]
        
        # Combine location impact
        if location:
            total_location_impact = sum(c['contribution_euro'] for c in location)
            combined_location = {
                'feature': 'Location',
                'raw_name': 'location_combined',
                'value': None,
                'importance': sum(c['importance'] for c in location),
                'contribution_euro': total_location_impact,
                'contribution_log': sum(c['contribution_log'] for c in location)
            }
            if abs(total_location_impact) >= 1.0:
                non_location.append(combined_location)
        
        # Sort and return top N
        non_location.sort(key=lambda x: abs(x['contribution_euro']), reverse=True)
        return non_location[:top_n]


def format_confidence_level(confidence_score: float) -> str:
    """
    Convert confidence score to human-readable level.
    
    Args:
        confidence_score: Score between 0-100
        
    Returns:
        Confidence level string
    """
    # NOTE: The bucket thresholds are intentionally asymmetric.
    # "Very High" is harder to reach (≥90) than "Very Low" (<40) because
    # confidence scores tend to cluster in the 40-75 range for this model.
    # This asymmetry ensures users see meaningful differentiation at the
    # upper end while avoiding over-classification as "Very Low".
    if confidence_score >= 90:
        return "Very High"
    elif confidence_score >= 75:
        return "High"
    elif confidence_score >= 60:
        return "Moderate"
    elif confidence_score >= 40:
        return "Low"
    else:
        return "Very Low"


def format_stability_level(stability_score: float) -> str:
    """Describe perturbation stability without implying statistical confidence."""
    if stability_score >= 90:
        return "Very Stable"
    if stability_score >= 75:
        return "Stable"
    if stability_score >= 60:
        return "Moderately Stable"
    if stability_score >= 40:
        return "Sensitive"
    return "Very Sensitive"


def format_contribution_text(contribution: Dict) -> str:
    """
    Format a feature contribution as readable text.
    
    Args:
        contribution: Feature contribution dictionary
        
    Returns:
        Formatted string describing the contribution
    """
    feature = contribution['feature']
    euro_impact = contribution['contribution_euro']
    
    if abs(euro_impact) < 0.01:
        return f"{feature}: no impact"
    elif abs(euro_impact) < 1:
        return f"{feature}: minimal impact"
    
    direction = "higher" if euro_impact > 0 else "lower"
    return f"{feature}: estimate €{abs(euro_impact):.0f} {direction} vs reference"
