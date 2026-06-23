"""
Prediction utilities for rent price estimation.
Provides confidence scores and feature importance explanations.
"""
import numpy as np
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')


class PredictionAnalyzer:
    """Analyzes predictions and provides confidence scores and feature explanations."""
    
    def __init__(self, model, feature_medians=None):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            model: Trained XGBRegressor or RandomForestRegressor model
            feature_medians: Optional array of training feature medians for baseline
                            computation. If not provided, uses 0 as baseline for
                            continuous features (simplified approach).
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
        Calculate prediction confidence score based on model uncertainty.
        
        For Random Forest: Uses standard deviation of individual tree predictions.
        For XGBoost: Uses quantile prediction approach or feature-based uncertainty.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound, confidence_score)
            confidence_score is between 0-100, where 100 is highest confidence
        """
        # Ensure single-row input
        features = np.atleast_2d(features)
        if features.shape[0] > 1:
            features = features[:1]  # Use only first row if multiple provided
        
        # Get base prediction
        mean_prediction = self.model.predict(features)[0]
        
        if self.is_random_forest:
            # Random Forest: Use tree predictions variance
            tree_predictions = np.array([tree.predict(features)[0] 
                                        for tree in self.model.estimators_])
            std_prediction = np.std(tree_predictions)
        
        elif self.is_xgboost:
            # XGBoost: Estimate uncertainty using feature perturbation
            # Apply small random perturbations and measure prediction variance
            n_samples = 30
            perturbations = []
            
            # Deterministic seed based on input hash for reproducibility
            seed = int(abs(np.sum(features * 1000)) % 10000)
            rng = np.random.RandomState(seed)
            
            for _ in range(n_samples):
                # Create small random noise (±5% of feature values)
                noise = rng.normal(0, 0.05, features.shape)
                perturbed_features = features + (features * noise)
                # Keep binary features as 0 or 1
                for idx in range(features.shape[1]):
                    if features[0, idx] in [0, 1]:
                        perturbed_features[0, idx] = features[0, idx]
                
                pred = self.model.predict(perturbed_features)[0]
                perturbations.append(pred)
            
            std_prediction = np.std(perturbations)
        
        else:
            # Unknown model type: use conservative estimate
            std_prediction = abs(mean_prediction) * 0.15
        
        # Calculate prediction interval (95% confidence interval)
        lower_bound = mean_prediction - 1.96 * std_prediction
        upper_bound = mean_prediction + 1.96 * std_prediction
        
        # Calculate confidence score based on relative prediction interval width
        if mean_prediction != 0:
            interval_width = upper_bound - lower_bound
            # Add epsilon to prevent division by near-zero overflow
            relative_interval = interval_width / max(abs(mean_prediction), 1e-10)
            
            # Map relative interval to confidence score
            confidence_score = 100 / (1 + np.exp(5 * (relative_interval - 0.5)))
            confidence_score = np.clip(confidence_score, 0, 100)
            # Replace any NaN values (from overflow/underflow) with midpoint confidence
            confidence_score = np.nan_to_num(confidence_score, nan=50.0)
        else:
            confidence_score = 50.0
        
        # Convert to Python native types for Streamlit compatibility
        return float(mean_prediction), float(lower_bound), float(upper_bound), float(confidence_score)
    
    def get_feature_contributions(self, features: np.ndarray, base_prediction: float) -> List[Dict]:
        """
        Calculate individual feature contributions to the prediction.
        
        Uses a perturbation approach: measures how prediction changes when each
        feature is set to its mean training value.
        
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
        
        # Calculate baseline values for feature perturbation
        # NOTE: Using 0 as baseline for continuous features and 0.5 for binary features.
        # This means "what would the prediction be if this feature were 0/neutral".
        # While not as accurate as true training data medians, it provides a valid
        # reference point for measuring feature impact. For better accuracy, pass
        # feature_medians to the constructor.
        baseline_values = np.zeros(len(self.feature_names))
        for idx, val in enumerate(features_flat):
            if val in [0, 1]:
                # Binary feature: use 0.5 as neutral baseline
                baseline_values[idx] = 0.5
            elif self.feature_medians is not None and idx < len(self.feature_medians):
                # Use provided training median if available
                baseline_values[idx] = self.feature_medians[idx]
            else:
                # Continuous feature: use 0 as baseline (simplified approach)
                baseline_values[idx] = 0.0
        
        # Calculate contribution for each feature
        for idx, feature_name in enumerate(self.feature_names):
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
            'condition_ottimo / ristrutturato': 'Excellent Condition'
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
                        if not (c['raw_name'].startswith('latitude') or 
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
    
    direction = "adds" if euro_impact > 0 else "reduces"
    return f"{feature} {direction} €{abs(euro_impact):.0f}"
