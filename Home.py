import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from config import (
    DATA_AS_OF,
    ENERGY_CLASSES,
    GEOGRAPHIC_MAPS_ENABLED,
    ITALY_CENTER_COORDS,
    MAX_AREA_M2,
    MAX_BATHROOMS,
    MAX_ROOMS,
    MIN_AREA_M2,
    MIN_BATHROOMS,
    MIN_ROOMS,
    PHOTON_API_BASE_URL,
    PHOTON_COUNTRY_CODE,
)
from prediction_utils import (
    PredictionAnalyzer,
    format_contribution_text,
    format_stability_level,
)
from app_state import (
    build_prediction_snapshot,
    get_current_location,
    store_resolved_location,
    sync_location_query,
)
from map_data import (
    load_neighborhood_price_data, 
    load_property_cluster_data,
    get_price_category
)
from feature_utils import (
    find_similar_properties,
    calculate_affordability,
    generate_prediction_report
)


@st.cache_resource(ttl=3600)
def load_model(path='rent_prediction_model/rent_model_v2.pkl'):
    """Load the trained model with comprehensive error handling."""
    try:
        model = joblib.load(path)
        # Verify model has required attributes
        if not hasattr(model, 'predict'):
            st.error(f"Error: Loaded object from {path} is not a valid model (missing predict method).")
            return None
        return model
    except FileNotFoundError:
        st.error(f"❌ Error: Model file not found at {path}")
        st.info("💡 Please ensure the model file exists in the correct location.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.info("💡 The model file may be corrupted. Please check the file integrity.")
        return None

@st.cache_resource(ttl=3600)
def load_prediction_analyzer():
    """Initialize the prediction analyzer with the loaded model."""
    model = load_model()
    if model is not None:
        try:
            return PredictionAnalyzer(model)
        except Exception as e:
            st.error(f"❌ Error initializing prediction analyzer: {e}")
            return None
    return None
    
model = load_model()
analyzer = load_prediction_analyzer()

# Issue #5 fix: Graceful degradation if model fails to load
if model is None or analyzer is None:
    st.error("🚨 Critical Error: Unable to load the prediction model. The application cannot process predictions.")
    st.info("ℹ️ Please contact the system administrator or check the model file.")
# Initialize session state for predictions (only relevant when model is loaded)
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

st.title('Predict Rent Prices in Italy 🇮🇹', anchor=False)
st.caption(
    f"Historical advertised-rent estimate · data through {DATA_AS_OF} · not a live market quote"
)


def get_binary_value(option):
    """Convert 'Yes'/'No' to binary 1/0."""
    return 1 if option == 'Yes' else 0

def calculate_rooms_per_area(rooms, area):
    return rooms / area if area > 0 else 0

def calculate_baths_per_room(bathrooms, rooms):
    return bathrooms / rooms if rooms > 0 else 0

def calculate_amenity_score(balcony, fiber_optic, electric_gate, shared_garden, external_exposure):
    return balcony + fiber_optic + electric_gate + shared_garden + external_exposure

def predict_rent(features):
    """
    Make rent prediction with comprehensive error handling.
    
    Args:
        features: Feature array for prediction
        
    Returns:
        Tuple of (euro_est, perturbation_lower, perturbation_upper,
        stability_score, sensitivity_items)
        or (None, None, None, None, None) if prediction fails
    """
    if model is None or analyzer is None:
        st.error("❌ Model is not loaded. Cannot perform prediction.")
        return None, None, None, None, None
    
    try:
        # Validate feature array
        if features is None or len(features) == 0:
            st.error("❌ Invalid feature array provided.")
            return None, None, None, None, None
        
        # Input perturbation diagnostics; these are not calibrated uncertainty.
        log_estimate, log_lower, log_upper, stability_score = analyzer.calculate_confidence_score(features)
        
        # Convert from log space to euro
        euro_est = np.expm1(log_estimate)
        lower_bound = np.expm1(log_lower)
        upper_bound = np.expm1(log_upper)
        
        # Ensure bounds are non-negative and logical
        lower_bound = max(0, lower_bound)
        upper_bound = max(lower_bound, upper_bound)
        
        # Calculate feature contributions
        contributions = analyzer.get_feature_contributions(features, log_estimate)
        top_contributors = analyzer.get_top_contributors(contributions, top_n=5)
        
        return euro_est, lower_bound, upper_bound, stability_score, top_contributors
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.info("💡 Please check your input values and try again.")
        return None, None, None, None, None

    

def extract_location_hierarchy(place_properties):
    """
    Extract region, city, and neighborhood coordinates from Photon API response.
    Falls back to primary coordinates if specific levels are not available.
    
    Args:
        place_properties: Properties dict from Photon API response
        
    Returns:
        Tuple of (state, city, neighborhood) as strings
    """
    # Extract available location information
    state = place_properties.get('state', '')  # Region level
    city = place_properties.get('city', '') or place_properties.get('county', '')
    neighborhood = place_properties.get('district', '') or place_properties.get('suburb', '') or place_properties.get('locality', '')
    
    return state, city, neighborhood

def geocode_location_level(location_name, country='Italy'):
    """
    Geocode a specific location to get its coordinates.
    
    Args:
        location_name: Name of the location to geocode
        country: Country name (default: Italy)
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if not found
    """
    if not location_name:
        return None, None
    
    try:
        headers = {'User-Agent': 'ItalyRentPrediction/1.0 (https://github.com/aryanahamed/italy-rent-prediction)'}
        response = requests.get(
            PHOTON_API_BASE_URL,
            params={
                'q': f"{location_name}, {country}",
                'limit': 1,
                'countrycode': PHOTON_COUNTRY_CODE,
                'lang': 'it',
            },
            timeout=5,
            headers=headers,
        )
        if response.status_code == 200:
            features = response.json().get('features', [])
            if features:
                coords = features[0]['geometry']['coordinates']
                return coords[1], coords[0]  # lat, lon
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pass
    
    return None, None

if model is not None and analyzer is not None:
    address = st.text_input(
        'Search for an Italian neighborhood or address',
        key='address',
        help="Example: 'Navigli, Milano'. Your query is sent to the Photon geocoding service."
        )

    sync_location_query(st.session_state, address)

    # Issue #5 fix: Empty address validation
    if address and address.strip():
        headers = {'User-Agent': 'ItalyRentPrediction/1.0 (https://github.com/aryanahamed/italy-rent-prediction)'}
        try:
            response = requests.get(
                PHOTON_API_BASE_URL,
                params={
                    'q': address,
                    'limit': 5,
                    'countrycode': PHOTON_COUNTRY_CODE,
                    'lang': 'it',
                },
                timeout=10,
                headers=headers,
            )
            response.raise_for_status()
            suggestions = [
                place for place in response.json().get('features', [])
                if place.get('properties', {}).get('countrycode', '').upper()
                == PHOTON_COUNTRY_CODE
            ]

            if suggestions:
                # Create unique options by adding more context and removing duplicates
                options = []
                place_mapping = {}  # Map option label to place index
                seen = set()

                for idx, place in enumerate(suggestions):
                    props = place['properties']
                    # Build a more detailed label
                    name = props.get('name', 'Unknown')
                    city = props.get('city', '')
                    state = props.get('state', '')
                    country = props.get('country', 'Italy')

                    # Create label with available info
                    if city and state:
                        label = f"{name}, {city}, {state}, {country}"
                    elif city:
                        label = f"{name}, {city}, {country}"
                    elif state:
                        label = f"{name}, {state}, {country}"
                    else:
                        label = f"{name}, {country}"

                    # Only add if not duplicate
                    if label not in seen:
                        options.append(label)
                        place_mapping[label] = idx
                        seen.add(label)

                if not options:
                    options = [f"{place['properties'].get('name', 'Unknown')}, {place['properties'].get('country', 'Italy')}" for place in suggestions]
                    place_mapping = {label: idx for idx, label in enumerate(options)}

                selected_option = st.selectbox(
                    "Select a confirmed location",
                    options,
                    index=None,
                    placeholder="Choose one result…",
                    key="selected_location",
                )

                if selected_option is not None:
                    # Get the correct place using the mapping
                    selected_place = suggestions[place_mapping[selected_option]]
                    lat = selected_place['geometry']['coordinates'][1]
                    lon = selected_place['geometry']['coordinates'][0]

                    # Extract location hierarchy
                    properties = selected_place['properties']
                    state, city, neighborhood = extract_location_hierarchy(properties)

                    region_lat, region_lon = geocode_location_level(state) if state else (lat, lon)
                    city_lat, city_lon = geocode_location_level(city) if city else (lat, lon)
                    region_coords = (
                        region_lat if region_lat is not None else lat,
                        region_lon if region_lon is not None else lon,
                    )
                    city_coords = (
                        city_lat if city_lat is not None else lat,
                        city_lon if city_lon is not None else lon,
                    )
                    place_id = f"{properties.get('osm_type', '')}:{properties.get('osm_id', '')}"
                    store_resolved_location(
                        st.session_state,
                        query=address,
                        display_label=selected_option,
                        latitude=lat,
                        longitude=lon,
                        region=state,
                        city=city,
                        neighborhood=neighborhood or properties.get('name', ''),
                        region_coordinates=region_coords,
                        city_coordinates=city_coords,
                        country_code=properties.get('countrycode', PHOTON_COUNTRY_CODE),
                        place_id=place_id,
                    )
                    st.map({'latitude': [lat], 'longitude': [lon]})
            else:
                st.write("No suggestions found. Please refine your search.")
        except requests.exceptions.ConnectTimeout:
            st.error("The location search timed out. Please retry or refine the query.")
        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to the location service. Please retry later.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error fetching location suggestions: {str(e)}")
            st.info("A confirmed Italian location is required before prediction.")

    with st.form(key='rent_form'):

        st.subheader('Property Details', anchor=False)

        col1, col2 = st.columns(2)
        with col1:
            rooms = st.number_input(
                'Rooms', min_value=MIN_ROOMS, max_value=MAX_ROOMS, value=2, key='rooms'
            )
            area = st.number_input(
                'Area (m²)', min_value=MIN_AREA_M2, max_value=MAX_AREA_M2, value=80, key='area'
            )
            monthly_household_income = st.number_input(
                'Monthly disposable household income (optional)',
                min_value=0,
                value=0,
                step=100,
                key='monthly_household_income',
                help='Used only for a transparent rent-to-income ratio. Enter 0 to omit.',
            )

        with col2:
            bathrooms = st.number_input(
                'Bathrooms', min_value=MIN_BATHROOMS, max_value=MAX_BATHROOMS,
                value=1, key='bathrooms'
            )
            selected_energy_class = st.selectbox('Energy Class',
                                                 options=ENERGY_CLASSES,
                                                 key='energy_class',
                                                 index=0)

        st.divider()

        col3, col4, col5 = st.columns(3)
        with col3:
            st.caption("Amenities · choose Yes only when known present")
            parking_spots = st.radio('Parking spots', options=['No', 'Yes'], key='parking_spots', horizontal=True)
            balcony = st.radio('Balcony', options=['No', 'Yes'], key='balcony', horizontal=True)
            shared_garden = st.radio('Shared Garden', options=['No', 'Yes'], key='shared_garden', horizontal=True)
            cellar = st.radio('Cellar', options=['No', 'Yes'], key='cellar', horizontal=True)
            pool = st.radio('Pool', options=['No', 'Yes'], key='pool', horizontal=True)

        with col4:
            st.caption("Features · choose Yes only when known present")
            furnished = st.radio('Furnished', options=['No', 'Yes'], key='furnished', horizontal=True)
            fiber_optic = st.radio('Fiber Optic', options=['No', 'Yes'], key='fiber_optic', horizontal=True)
            electric_gate = st.radio('Electric Gate', options=['No', 'Yes'], key='electric_gate', horizontal=True)
            central_heating = st.radio('Central Heating', options=['No', 'Yes'], key='central_heating', horizontal=True)
            sea_view = st.radio('Sea View', options=['No', 'Yes'], key='sea_view', horizontal=True)

        with col5:
            st.caption("Condition & Layout")
            condition = st.selectbox(
                'Condition',
                options=['Unknown / Not provided', 'Good/Habitable',
                         'Excellent/Renovated', 'New/Under Construction',
                         'To be Renovated'],
                key='condition',
            )
            top_floor = st.radio('Top Floor', options=['No', 'Yes'], key='top_floor', horizontal=True)
            external_exposure = st.radio('External Exposure', options=['No', 'Yes'], key='external_exposure', horizontal=True)

        submit_button = st.form_submit_button(label='Predict Rent', type="primary")

    if submit_button:
        resolved_location = get_current_location(st.session_state, address)
        if not address or not address.strip():
            st.error("⚠️ Please enter an address in the search field above before making a prediction.")
        elif resolved_location is None:
            st.error("Please select a confirmed Italian location for the current search before predicting.")
        else:
            lat_neigh = resolved_location['latitude_neighborhood']
            lon_neigh = resolved_location['longitude_neighborhood']
            lat_city = resolved_location['latitude_city']
            lon_city = resolved_location['longitude_city']
            lat_region = resolved_location['latitude_region']
            lon_region = resolved_location['longitude_region']

            input_snapshot = {
                'rooms': int(rooms),
                'area_m2': int(area),
                'bathrooms': int(bathrooms),
                'energy_class': selected_energy_class,
                'parking_spots': parking_spots,
                'balcony': balcony,
                'shared_garden': shared_garden,
                'cellar': cellar,
                'pool': pool,
                'furnished': furnished,
                'fiber_optic': fiber_optic,
                'electric_gate': electric_gate,
                'central_heating': central_heating,
                'sea_view': sea_view,
                'condition': condition,
                'top_floor': top_floor,
                'external_exposure': external_exposure,
                'monthly_household_income': (
                    float(monthly_household_income)
                    if monthly_household_income > 0 else None
                ),
            }

            parking_spots = get_binary_value(parking_spots)
            balcony = get_binary_value(balcony)
            fiber_optic = get_binary_value(fiber_optic)
            shared_garden = get_binary_value(shared_garden)
            cellar = get_binary_value(cellar)
            furnished = get_binary_value(furnished)
            external_exposure = get_binary_value(external_exposure)
            electric_gate = get_binary_value(electric_gate)
            top_floor = get_binary_value(top_floor)
            central_heating = get_binary_value(central_heating)
            sea_view = get_binary_value(sea_view)
            pool = get_binary_value(pool)

            # One-Hot Encoding for Energy Class (Reference: A)
            ec_B = 1 if selected_energy_class == 'B' else 0
            ec_C = 1 if selected_energy_class == 'C' else 0
            ec_D = 1 if selected_energy_class == 'D' else 0
            ec_E = 1 if selected_energy_class == 'E' else 0
            ec_F = 1 if selected_energy_class == 'F' else 0
            ec_G = 1 if selected_energy_class == 'G' else 0
            ec_Unknown = 1 if selected_energy_class == 'Unknown' else 0

            # One-Hot Encoding for Condition
            cond_buono = 1 if condition == 'Good/Habitable' else 0
            cond_da_rist = 1 if condition == 'To be Renovated' else 0
            cond_nuovo = 1 if condition == 'New/Under Construction' else 0
            cond_ottimo = 1 if condition == 'Excellent/Renovated' else 0

            furnished_and_central_heating = 1 if furnished == 1 and central_heating == 1 else 0

            # Calculate derived features for new model
            # Note: area is now passed as raw value

            # 1. Rooms per Area
            rooms_per_area = calculate_rooms_per_area(rooms, area)

            # 2. Baths per Room
            baths_per_room = calculate_baths_per_room(bathrooms, rooms)

            # 3. Amenity Score
            amenity_score = calculate_amenity_score(balcony, fiber_optic, electric_gate, shared_garden, external_exposure)

            # Build feature array matching EXACT model feature order (36 features)
            features = np.array([[
                parking_spots, bathrooms, rooms, top_floor, sea_view, central_heating,
                area, furnished, balcony, external_exposure, fiber_optic, electric_gate,
                cellar, shared_garden, pool,
                ec_B, ec_C, ec_D, ec_E, ec_F, ec_G, ec_Unknown,
                furnished_and_central_heating,
                lat_region, lon_region, lat_city, lon_city, lat_neigh, lon_neigh,
                cond_buono, cond_da_rist, cond_nuovo, cond_ottimo,
                rooms_per_area, baths_per_room, amenity_score
            ]])

            # Show processing indicator
            with st.spinner("Calculating prediction..."):
                # Predict rent and calculate local stability/sensitivity diagnostics.
                result = predict_rent(features)

            if result[0] is not None:
                euro_est, lower_bound, upper_bound, stability_score, sensitivity_items = result
                st.session_state.prediction_results = build_prediction_snapshot(
                    location=resolved_location,
                    inputs=input_snapshot,
                    estimate=euro_est,
                    perturbation_lower=lower_bound,
                    perturbation_upper=upper_bound,
                    stability_score=stability_score,
                    sensitivity_items=sensitivity_items,
                    data_as_of=DATA_AS_OF,
                )

    # Display prediction results if they exist in session state
    if st.session_state.prediction_results is not None:
        results = st.session_state.prediction_results
        euro_est = results['euro_est']
        lower_bound = results['lower_bound']
        upper_bound = results['upper_bound']
        stability_score = results['stability_score']
        top_contributors = results['sensitivity_items']

        # Display main prediction
        st.success("Prediction successful!")
        st.header(f'Predicted Rent Price: €{euro_est:.0f}', anchor=False)

        # This is a perturbation diagnostic, not a probability of correctness.
        stability_level = format_stability_level(stability_score)
        col_conf1, col_conf2 = st.columns([2, 1])

        with col_conf1:
            st.subheader(f"Input Stability: {stability_level}", anchor=False)
            stability_value = stability_score / 100 if np.isfinite(stability_score) else 0
            st.progress(max(0.0, min(1.0, stability_value)))

        with col_conf2:
            st.metric("Stability Score", f"{stability_score:.1f}/100")

        st.subheader(f"Input Perturbation Range: €{lower_bound:.0f} - €{upper_bound:.0f}", anchor=False)
        range_width = upper_bound - lower_bound
        st.caption(
            f"Range width: €{range_width:.0f}. This diagnostic varies area by ±5% and "
            "rooms/bathrooms by one within supported bounds; "
            "it is not a confidence interval or a probability that the actual rent falls in this range."
        )

        st.divider()

        st.subheader("🎯 One-Feature Model Sensitivity", anchor=False)
        st.caption("How the model output changes when one supported feature is changed to its reference value.")

        for i, contrib in enumerate(top_contributors, 1):
            contribution_text = format_contribution_text(contrib)
            impact = contrib['contribution_euro']

            direction_icon = "↗" if impact > 0 else "↘" if impact < 0 else "•"

            # Display with visual indicator - single row
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{direction_icon} **{i}.** {contribution_text}")
            with col2:
                st.write(f"**€{abs(impact):.0f}**")
            with col3:
                direction = "↑" if impact > 0 else "↓"
                st.write(f"{direction} {abs(impact/euro_est*100):.1f}%")

        st.caption(f"📍 Based on location: {results['address']}")
        st.caption("These checks are model sensitivities, not causal effects or an additive price breakdown.")

        # Add explanation tooltip
        with st.expander("ℹ️ How to interpret these results"):
            st.markdown("""
            **Input Stability Score:** Summarizes how much the prediction moves when numeric inputs are
            changed locally (area ±5%, rooms/bathrooms ±1). A high value means the prediction was
            locally stable in this diagnostic only.

            **Input Perturbation Range:** The observed model-output range across those synthetic changes.
            It is not a calibrated uncertainty interval and does not describe a 95% probability.

            **One-Feature Sensitivity:** Compares the submitted input with a reference setting while keeping
            other inputs fixed. Binary and encoded-category flags use zero as their stated reference. These values are
            model-specific associations, not causes, and they are not expected to sum to the prediction.
            """)

        st.divider()

        # ========== FEATURE #1: AFFORDABILITY CALCULATOR ==========
        st.subheader("💰 Affordability Analysis", anchor=False)

        location_snapshot = results['location']
        input_snapshot = results['inputs']
        region_name = location_snapshot.get('region')
        city_name = location_snapshot.get('city')

        # Display location context if available
        if region_name:
            st.caption(f"📍 Region: {region_name}" + (f" | City: {city_name}" if city_name else ""))

        household_income = input_snapshot.get('monthly_household_income')
        affordability = calculate_affordability(euro_est, household_income)
        st.info(
            "The 30% rent-to-income ratio is a simple reference point, not a personalized financial verdict. "
            "It does not include utilities, taxes, debt, household size, or savings goals."
        )

        col_aff1, col_aff2, col_aff3 = st.columns(3)

        with col_aff1:
            st.metric(
                "30% Reference",
                affordability['affordability_level'],
                delta=None
            )
            st.caption("Based only on the income entered for this prediction")

        with col_aff2:
            st.metric(
                "Required Monthly Income",
                f"€{affordability['required_income']:.0f}",
                delta=None
            )
            st.caption("To afford this rent comfortably (30% rule)")

        with col_aff3:
            if household_income:
                ratio = affordability['pct_of_income']
                st.metric("Submitted Rent-to-Income", f"{ratio:.1f}%")
                st.caption(f"Income entered: €{household_income:.0f}/month")
            else:
                st.metric("Submitted Rent-to-Income", "Not calculated")
                st.caption("Enter household income in the form to calculate it")

        if household_income:
            ratio = affordability['pct_of_income']
            st.progress(max(0.0, min(1.0, ratio / 100)))
            if ratio <= 30:
                st.success("This rent is at or below the 30% reference for the submitted income.")
            else:
                st.warning(f"This rent is {ratio - 30:.1f} percentage points above the 30% reference.")
        else:
            st.info("No income was provided, so no affordability classification is shown.")

        st.divider()

        # ========== FEATURE #3: SIMILAR PROPERTIES ==========
        st.subheader("🏘️ Similar Properties in the Area", anchor=False)
        st.caption("Archived same-city asking-rent records matched on rooms and area; predicted price is not used for matching.")

        with st.spinner("Finding similar properties..."):
            input_rooms = input_snapshot['rooms']
            input_area = input_snapshot['area_m2']
            try:
                similar_props = find_similar_properties(
                    city=city_name,
                    rooms=input_rooms,
                    area=input_area,
                    top_n=5,
                ) if city_name else []
            except RuntimeError as exc:
                st.error(f"Comparable records could not be loaded: {exc}")
                similar_props = []

        if similar_props:
            # Calculate price comparison
            similar_prices = [p['price'] for p in similar_props]
            avg_similar_price = np.mean(similar_prices)
            min_similar_price = min(similar_prices)
            max_similar_price = max(similar_prices)

            col_sim1, col_sim2, col_sim3 = st.columns(3)
            with col_sim1:
                st.metric("Your Prediction", f"€{euro_est:.0f}")
            with col_sim2:
                diff = avg_similar_price - euro_est
                st.metric("Similar Props Avg", f"€{avg_similar_price:.0f}",
                         delta=f"{diff:+.0f}" if abs(diff) > 1 else "Similar",
                         delta_color="inverse")
            with col_sim3:
                st.metric("Range", f"€{min_similar_price:.0f} - €{max_similar_price:.0f}")

            # Display similar properties in a nice format
            for i, prop in enumerate(similar_props, 1):
                with st.expander(f"🏠 {i}. {prop['neighborhood']}, {prop['city']} - €{prop['price']:.0f}/month"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Property Details:**")
                        st.write(f"• Rooms: {prop['rooms']}")
                        st.write(f"• Area: {prop['area']:.0f} m²")
                        st.write(f"• Bathrooms: {prop['bathrooms']}")
                        st.write(f"• Energy Class: {prop['energy_class']}")

                    with col2:
                        st.write("**Amenities:**")
                        st.write(f"• Furnished: {prop['furnished']}")
                        st.write(f"• Balcony: {prop['balcony']}")
                        st.write(f"• Parking: {prop['parking']}")
                        st.write(f"• Condition: {prop['condition']}")

                    with col3:
                        st.write("**Price Comparison:**")
                        price_diff = prop['price'] - euro_est
                        if abs(price_diff) < 50:
                            st.write(f"💚 Very similar: €{prop['price']:.0f}")
                        elif price_diff > 0:
                            st.write(f"🔴 Pricier: +€{abs(price_diff):.0f}")
                        else:
                            st.write(f"🔵 Cheaper: -€{abs(price_diff):.0f}")
                        st.write(f"• Match: {prop['similarity_score']*100:.0f}%")
        else:
            st.info("No similar properties found in our database for this search criteria.")

        st.divider()

        st.subheader("📈 Historical Price Trends", anchor=False)
        st.info(
            "Unavailable for this dataset. The source is a dated listing snapshot, not repeated observations "
            "over time, so treating its rows as a time series would create a false market trend."
        )
        hist_stats = None

        st.divider()

        # ========== FEATURE #4: PROPERTY FEATURE OPTIMIZER ==========
        st.subheader("🎨 Compare a Property Scenario", anchor=False)
        st.caption("Change selected inputs and run a separate model estimate. This does not optimize a property or predict renovation returns.")

        with st.expander("🔧 Try Different Feature Combinations", expanded=False):
            st.write("**Adjust features below to see how they impact the predicted rent:**")

            current_parking = input_snapshot['parking_spots']
            current_balcony = input_snapshot['balcony']
            current_furnished = input_snapshot['furnished']
            current_fiber = input_snapshot['fiber_optic']
            current_heating = input_snapshot['central_heating']
            current_garden = input_snapshot['shared_garden']
            current_gate = input_snapshot['electric_gate']

            col_opt1, col_opt2, col_opt3 = st.columns(3)

            with col_opt1:
                st.write("**Toggle Amenities:**")
                opt_parking = st.checkbox("Add Parking", value=(current_parking == 'Yes'), key='opt_parking')
                opt_balcony = st.checkbox("Add Balcony", value=(current_balcony == 'Yes'), key='opt_balcony')
                opt_garden = st.checkbox("Add Shared Garden", value=(current_garden == 'Yes'), key='opt_garden')

            with col_opt2:
                st.write("**Toggle Features:**")
                opt_furnished = st.checkbox("Furnished", value=(current_furnished == 'Yes'), key='opt_furnished')
                opt_heating = st.checkbox("Central Heating", value=(current_heating == 'Yes'), key='opt_heating')
                opt_fiber = st.checkbox("Fiber Optic", value=(current_fiber == 'Yes'), key='opt_fiber')

            with col_opt3:
                st.write("**Toggle Extras:**")
                opt_gate = st.checkbox("Electric Gate", value=(current_gate == 'Yes'), key='opt_gate')
                opt_rooms_adj = st.slider("Adjust Rooms", min_value=MIN_ROOMS, max_value=MAX_ROOMS,
                                           value=input_snapshot['rooms'], key='opt_rooms')

            # Additional adjustable features (Issue #4 fix)
            st.write("**Adjust Core Features:**")
            col_adj1, col_adj2, col_adj3 = st.columns(3)

            with col_adj1:
                opt_bathrooms = st.slider("Bathrooms", min_value=MIN_BATHROOMS, max_value=MAX_BATHROOMS,
                                          value=input_snapshot['bathrooms'], key='opt_bathrooms')

            with col_adj2:
                opt_area_sqm = st.slider("Area (m²)", min_value=MIN_AREA_M2, max_value=MAX_AREA_M2,
                                         value=input_snapshot['area_m2'], key='opt_area_sqm')
                opt_area = float(opt_area_sqm)

            with col_adj3:
                current_energy_val = input_snapshot['energy_class']
                current_energy_idx = ENERGY_CLASSES.index(current_energy_val) if current_energy_val in ENERGY_CLASSES else len(ENERGY_CLASSES) // 2
                opt_energy_class = st.selectbox("Energy Class", options=ENERGY_CLASSES,
                                                index=current_energy_idx, key='opt_energy_class')

            # Calculate new prediction with adjusted features
            if st.button("🔄 Calculate New Price", type="primary", key='optimize_btn'):
                opt_external = input_snapshot['external_exposure']
                opt_cellar = input_snapshot['cellar']
                opt_top_floor = input_snapshot['top_floor']
                opt_condition = input_snapshot['condition']

                # Convert to binary
                new_parking = 1 if opt_parking else 0
                new_balcony = 1 if opt_balcony else 0
                new_furnished = 1 if opt_furnished else 0
                new_heating = 1 if opt_heating else 0
                new_fiber = 1 if opt_fiber else 0
                new_garden = 1 if opt_garden else 0
                new_gate = 1 if opt_gate else 0
                new_external = get_binary_value(opt_external)
                new_cellar = get_binary_value(opt_cellar)
                new_top_floor = get_binary_value(opt_top_floor)

                # Calculate derived features for new model

                # One-Hot Encoding for Energy Class
                new_ec_B = 1 if opt_energy_class == 'B' else 0
                new_ec_C = 1 if opt_energy_class == 'C' else 0
                new_ec_D = 1 if opt_energy_class == 'D' else 0
                new_ec_E = 1 if opt_energy_class == 'E' else 0
                new_ec_F = 1 if opt_energy_class == 'F' else 0
                new_ec_G = 1 if opt_energy_class == 'G' else 0
                new_ec_Unknown = 1 if opt_energy_class == 'Unknown' else 0

                # One-Hot Encoding for Condition
                new_cond_buono = 1 if opt_condition == 'Good/Habitable' else 0
                new_cond_da_rist = 1 if opt_condition == 'To be Renovated' else 0
                new_cond_nuovo = 1 if opt_condition == 'New/Under Construction' else 0
                new_cond_ottimo = 1 if opt_condition == 'Excellent/Renovated' else 0

                new_furnished_heating = 1 if new_furnished == 1 and new_heating == 1 else 0

                fixed_sea_view = get_binary_value(input_snapshot['sea_view'])
                fixed_pool = get_binary_value(input_snapshot['pool'])

                # Calculate all derived features for the new model

                # 1. Rooms per Area
                new_rooms_per_area = calculate_rooms_per_area(opt_rooms_adj, opt_area)

                # 2. Baths per Room
                new_baths_per_room = calculate_baths_per_room(opt_bathrooms, opt_rooms_adj)

                opt_lat_neigh = location_snapshot['latitude_neighborhood']
                opt_lon_neigh = location_snapshot['longitude_neighborhood']
                opt_lat_city = location_snapshot['latitude_city']
                opt_lon_city = location_snapshot['longitude_city']
                opt_lat_region = location_snapshot['latitude_region']
                opt_lon_region = location_snapshot['longitude_region']

                # 3. Amenity Score
                new_amenity_score = calculate_amenity_score(new_balcony, new_fiber, new_gate, new_garden, new_external)

                # Build feature array matching EXACT model feature order (36 features)
                new_features = np.array([[
                    new_parking, opt_bathrooms, opt_rooms_adj, new_top_floor, fixed_sea_view, new_heating,
                    opt_area, new_furnished, new_balcony, new_external, new_fiber, new_gate,
                    new_cellar, new_garden, fixed_pool,
                    new_ec_B, new_ec_C, new_ec_D, new_ec_E, new_ec_F, new_ec_G, new_ec_Unknown,
                    new_furnished_heating,
                    opt_lat_region, opt_lon_region, opt_lat_city, opt_lon_city, opt_lat_neigh, opt_lon_neigh,
                    new_cond_buono, new_cond_da_rist, new_cond_nuovo, new_cond_ottimo,
                    new_rooms_per_area, new_baths_per_room, new_amenity_score
                ]])

                # Predict
                new_result = predict_rent(new_features)

                if new_result[0] is not None:
                    new_price = new_result[0]
                    price_change = new_price - euro_est

                    st.success(f"✨ New Predicted Price: **€{new_price:.0f}/month**")

                    if abs(price_change) < 10:
                        st.info(f"No significant change from original prediction")
                    elif price_change > 0:
                        st.info(f"📈 Price increased by **€{price_change:.0f}** (+{(price_change/euro_est*100):.1f}%)")
                    else:
                        st.info(f"📉 Price decreased by **€{abs(price_change):.0f}** ({(price_change/euro_est*100):.1f}%)")

                    st.write("**Changed Inputs:**")
                    scenario_values = {
                        'Parking': 'Yes' if opt_parking else 'No',
                        'Balcony': 'Yes' if opt_balcony else 'No',
                        'Shared garden': 'Yes' if opt_garden else 'No',
                        'Furnished': 'Yes' if opt_furnished else 'No',
                        'Central heating': 'Yes' if opt_heating else 'No',
                        'Fiber optic': 'Yes' if opt_fiber else 'No',
                        'Electric gate': 'Yes' if opt_gate else 'No',
                        'Rooms': opt_rooms_adj,
                        'Bathrooms': opt_bathrooms,
                        'Area (m²)': opt_area,
                        'Energy class': opt_energy_class,
                    }
                    original_values = {
                        'Parking': input_snapshot['parking_spots'],
                        'Balcony': input_snapshot['balcony'],
                        'Shared garden': input_snapshot['shared_garden'],
                        'Furnished': input_snapshot['furnished'],
                        'Central heating': input_snapshot['central_heating'],
                        'Fiber optic': input_snapshot['fiber_optic'],
                        'Electric gate': input_snapshot['electric_gate'],
                        'Rooms': input_snapshot['rooms'],
                        'Bathrooms': input_snapshot['bathrooms'],
                        'Area (m²)': float(input_snapshot['area_m2']),
                        'Energy class': input_snapshot['energy_class'],
                    }
                    impacts = [
                        f"{label}: {original_values[label]} → {new_value}"
                        for label, new_value in scenario_values.items()
                        if new_value != original_values[label]
                    ]

                    if impacts:
                        for impact in impacts:
                            st.write(f"• {impact}")
                    else:
                        st.write("• No changes made")

        st.divider()

        # ========== FEATURE #5: DOWNLOADABLE REPORT ==========
        st.subheader("📥 Download Prediction Report", anchor=False)
        st.caption("Save your prediction results for future reference")

        # Generate report
        try:
            report_text = generate_prediction_report(
                prediction_data=results,
                similar_properties=similar_props if 'similar_props' in locals() and similar_props else [],
                affordability=affordability if 'affordability' in locals() else None,
                historical_stats=None,
            )
        except Exception:
            report_text = f"Rent Prediction Report\n{'='*40}\n\n"
            report_text += f"Predicted Rent: €{euro_est:.0f}/month\n"
            report_text += f"Input stability: {stability_score:.1f}/100\n"
            report_text += f"Input perturbation range: €{lower_bound:.0f} - €{upper_bound:.0f}\n"
            report_text += f"Location: {results.get('address', 'Unknown')}\n"

        # Download button
        col_dl1, col_dl2 = st.columns([2, 1])
        with col_dl1:
            safe_address = ''.join(
                character if character.isalnum() or character in ('-', '_') else '_'
                for character in results['address']
            )[:80]
            st.download_button(
                label="📄 Download Full Report (TXT)",
                data=report_text,
                file_name=f"rent_prediction_report_{safe_address}_{euro_est:.0f}EUR.txt",
                mime="text/plain",
                help="Download a comprehensive text report with all prediction details"
            )

        with col_dl2:
            st.caption(f"Report size: {len(report_text)} chars")



else:
    st.info("🔧 Prediction features are unavailable because the model could not be loaded.")

if not GEOGRAPHIC_MAPS_ENABLED:
    st.divider()
    st.header("📍 Geographic Rental Views", anchor=False)
    st.info(
        "Temporarily unavailable. The current location cache is keyed only by neighborhood name, "
        "which can merge different Italian places with the same name. These maps will return after "
        "coordinates are rebuilt with city and region context."
    )
    st.stop()

# NEIGHBORHOOD PRICE HEATMAP SECTION
st.divider()
st.header("📍 Neighborhood Price Heatmap", anchor=False)
st.markdown("""
Explore average rental prices across Italian neighborhoods. 
**Warmer colors** indicate higher average rents, **cooler colors** show more affordable areas.
""")

# Add toggle for map visibility
show_heatmap = st.checkbox("Show Interactive Heatmap", value=False, help="Load the neighborhood price heatmap")

if show_heatmap:
    with st.spinner("Loading neighborhood data..."):
        try:
            # Load heatmap data
            heatmap_data, stats, neighborhoods = load_neighborhood_price_data()
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Neighborhoods", f"{stats['neighborhood_count']}")
            with col2:
                st.metric("Avg Price", f"€{stats['avg_price']:.0f}/mo")
            with col3:
                st.metric("Min Price", f"€{stats['min_price']:.0f}/mo")
            with col4:
                st.metric("Max Price", f"€{stats['max_price']:.0f}/mo")
            
            # Create folium map
            center_lat, center_lon = ITALY_CENTER_COORDS
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap',
                control_scale=True
            )
            
            # Add heatmap layer
            HeatMap(
                heatmap_data,
                min_opacity=0.3,
                max_zoom=13,
                radius=25,
                blur=35,
                gradient={
                    0.0: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime',
                    0.7: 'yellow',
                    1.0: 'red'
                }
            ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
            
            # Add data table with search
            with st.expander("🔍 Search Neighborhoods"):
                search_term = st.text_input("Filter by neighborhood name", "")
                
                # Convert to DataFrame for display
                neighborhoods_df = pd.DataFrame(neighborhoods)
                
                if search_term:
                    neighborhoods_df = neighborhoods_df[
                        neighborhoods_df['neighborhood'].str.contains(search_term, case=False, na=False)
                    ]
                
                # Sort by average price
                neighborhoods_df = neighborhoods_df.sort_values('avg_price', ascending=False)
                
                # Format for display
                display_df = neighborhoods_df.copy()
                display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"€{x:.0f}")
                display_df = display_df.drop(columns=['normalized_price'])
                display_df.columns = ['Neighborhood', 'Latitude', 'Longitude', 'Avg Price/mo', 'Properties']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption(f"Showing {len(display_df)} of {stats['neighborhood_count']} neighborhoods")
        
        except Exception as e:
            st.error(f"Failed to load heatmap data: {e}")
            st.info("Please ensure the data files are in the correct location.")


# PROPERTY CLUSTER MAP SECTION
st.divider()
st.header("🏘️ Property Cluster Map", anchor=False)
st.markdown("""
Explore individual rental properties grouped by location and similarity. 
Clusters show the **number of properties** in each area, with **color-coded markers** by price range.
""")

# Add toggle for cluster map visibility
show_cluster_map = st.checkbox("Show Property Clusters", value=False, help="Load interactive property cluster map")

if show_cluster_map:
    # Filter controls
    st.subheader("🎛️ Filter Properties", anchor=False)
    
    col1, col2 = st.columns(2)
    with col1:
        price_range = st.slider(
            "Price Range (€/month)",
            min_value=0,
            max_value=5000,
            value=(0, 3000),
            step=100,
            help="Filter properties by monthly rent",
            key="cluster_price_range"
        )
    
    with col2:
        room_range = st.slider(
            "Number of Rooms",
            min_value=1,
            max_value=10,
            value=(1, 5),
            step=1,
            help="Filter properties by room count",
            key="cluster_room_range"
        )
    
    # Create a cache key based on filter values
    cache_key = f"properties_{price_range[0]}_{price_range[1]}_{room_range[0]}_{room_range[1]}"
    
    # Check if we need to reload data (filters changed)
    if 'cluster_cache_key' not in st.session_state or st.session_state.cluster_cache_key != cache_key:
        with st.spinner("Loading property data..."):
            try:
                # Load filtered property data
                properties = load_property_cluster_data(
                    price_min=price_range[0],
                    price_max=price_range[1],
                    rooms_min=room_range[0],
                    rooms_max=room_range[1]
                )
                # Store in session state
                st.session_state.cluster_properties = properties
                st.session_state.cluster_cache_key = cache_key
            except Exception as e:
                st.error(f"Failed to load cluster map data: {e}")
                st.info("Please ensure the data files are in the correct location.")
                properties = []
    else:
        # Use cached data from session state
        properties = st.session_state.cluster_properties
    
    # Display the data
    if not properties:
        st.warning("No properties found with the selected filters. Try adjusting the ranges.")
    else:
        # Display summary statistics
        total_properties = len(properties)
        avg_price = sum(p['price'] for p in properties) / total_properties
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Properties Found", f"{total_properties:,}")
        with col2:
            st.metric("Average Price", f"€{avg_price:.0f}/mo")
        with col3:
            # Count price categories
            price_categories = {}
            for prop in properties:
                category = get_price_category(prop['price'])['label']
                price_categories[category] = price_categories.get(category, 0) + 1
            most_common = max(price_categories.items(), key=lambda x: x[1])[0]
            st.metric("Most Common", most_common.split('(')[0].strip())
        
        # Create map with clusters
        center_lat, center_lon = ITALY_CENTER_COORDS
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Create marker cluster with custom settings
        marker_cluster = MarkerCluster(
            name='Property Clusters',
            overlay=True,
            control=True,
            icon_create_function=None
        )
        
        # Add markers for each property
        for prop in properties:
            category = get_price_category(prop['price'])
            
            # Create popup content with property details
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="margin-bottom: 10px; color: #1f77b4;">
                    {prop['neighborhood']}, {prop['city']}
                </h4>
                <hr style="margin: 5px 0;">
                <p style="font-size: 18px; font-weight: bold; color: #2ca02c; margin: 5px 0;">
                    €{prop['price']:.0f}/month
                </p>
                <p style="margin: 3px 0;"><b>Rooms:</b> {prop['rooms']}</p>
                <p style="margin: 3px 0;"><b>Area:</b> {prop['area']} m²</p>
                <p style="margin: 3px 0;"><b>Bathrooms:</b> {prop['bathrooms']}</p>
                <p style="margin: 3px 0;"><b>Energy Class:</b> {prop['energy_class']}</p>
                <p style="margin: 3px 0;"><b>Furnished:</b> {prop['furnished']}</p>
                <p style="margin: 3px 0;"><b>Balcony:</b> {prop['balcony']}</p>
                <p style="margin: 3px 0;"><b>Parking:</b> {prop['parking']}</p>
                <hr style="margin: 5px 0;">
                <p style="font-size: 11px; color: #666; margin: 3px 0;">
                    Category: {category['label']}
                </p>
            </div>
            """
            
            # Create marker with color based on price category
            folium.Marker(
                location=[prop['latitude'], prop['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"€{prop['price']:.0f} - {prop['rooms']} rooms",
                icon=folium.Icon(color=category['color'], icon=category['icon'], prefix='fa')
            ).add_to(marker_cluster)
        
        # Add cluster to map
        marker_cluster.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map
        st_folium(m, width=700, height=500, key="cluster_map")
        
        # Legend
        st.markdown("### 🎨 Price Category Legend")
        legend_cols = st.columns(5)
        
        categories_info = [
            ("green", "Budget", "< €500"),
            ("blue", "Affordable", "€500-€1000"),
            ("orange", "Mid-range", "€1000-€1500"),
            ("red", "Premium", "€1500-€2500"),
            ("purple", "Luxury", "€2500+")
        ]
        
        for i, (color, label, price_range_text) in enumerate(categories_info):
            with legend_cols[i]:
                # Count properties in this category
                count = sum(1 for p in properties if get_price_category(p['price'])['color'] == color)
                
                # Color indicator (using emoji approximation)
                color_emoji = {
                    'green': '🟢',
                    'blue': '🔵', 
                    'orange': '🟠',
                    'red': '🔴',
                    'purple': '🟣'
                }
                
                st.markdown(f"""
                **{color_emoji.get(color, '⬜')} {label}**  
                {price_range_text}  
                *{count} properties*
                """)
        
        # Additional insights
        with st.expander("💡 Cluster Map Tips"):
            st.markdown("""
            **How to use this map:**
            - **Zoom in/out** to see individual properties or broader areas
            - **Click clusters** (numbered circles) to zoom into that area
            - **Click markers** to see detailed property information
            - **Adjust filters** above to narrow down your search
            - **Colors indicate price ranges** - see legend below map
            
            **Cluster numbers** represent how many properties are grouped in that location.
            The map automatically groups nearby properties for better visualization.
            """)




