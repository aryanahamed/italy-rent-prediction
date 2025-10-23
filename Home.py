import time
import streamlit as st
import joblib
import numpy as np
import requests
from config import ENERGY_CLASSES, ENERGY_CLASS_MAP, MARGIN_OF_ERROR
from prediction_utils import (
    PredictionAnalyzer,
    format_confidence_level,
    format_contribution_text
)


@st.cache_resource(ttl=3600)
def load_model(path='rent_prediction_model/rent_model_v2.pkl'):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

@st.cache_resource(ttl=3600)
def load_prediction_analyzer():
    """Initialize the prediction analyzer with the loaded model."""
    model = load_model()
    if model is not None:
        return PredictionAnalyzer(model)
    return None
    
model = load_model()
analyzer = load_prediction_analyzer()

st.title('Predict Rent Prices in Italy 🇮🇹', anchor=False)


def get_binary_value(option):
    """Convert 'Yes'/'No' to binary 1/0."""
    return 1 if option == 'Yes' else 0

def calculate_building_layout(cellar, top_floor):
    """Calculate building layout feature."""
    if cellar and top_floor:
        return 3
    elif cellar:
        return 1
    elif top_floor:
        return 2
    return 0

def predict_rent(features):
    if model is None or analyzer is None:
        st.error("Model is not loaded. Cannot perform prediction.")
        return None, None, None, None, None
    
    # Get confidence-based prediction with intervals
    log_estimate, log_lower, log_upper, confidence_score = analyzer.calculate_confidence_score(features)
    
    # Convert from log space to euro
    euro_est = np.expm1(log_estimate)
    lower_bound = np.expm1(log_lower)
    upper_bound = np.expm1(log_upper)
    
    # Ensure bounds are non-negative
    lower_bound = max(0, lower_bound)
    upper_bound = max(lower_bound, upper_bound)
    
    # Calculate feature contributions
    contributions = analyzer.get_feature_contributions(features, log_estimate)
    top_contributors = analyzer.get_top_contributors(contributions, top_n=5)
    
    return euro_est, lower_bound, upper_bound, confidence_score, top_contributors

    

address = st.text_input(
    'Search your desired neighborhood (For better results add city with the name)', 
    key='address',
    help="Example: 'Navigli, Milano'"
    )

if address:
    photon_url = f"https://photon.komoot.io/api/?q={address} Italy&limit=5"
    try:
        response = requests.get(photon_url, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 200:
            suggestions = response.json()['features']
            
            if suggestions:
                options = [f"{place['properties']['name']}, {place['properties']['country']}" for place in suggestions]
                selected_option = st.selectbox("Select an option", options)
                
                selected_place = suggestions[options.index(selected_option)]
                lat = selected_place['geometry']['coordinates'][1]
                lon = selected_place['geometry']['coordinates'][0]

                st.session_state['latitude'] = lat
                st.session_state['longitude'] = lon
                
                map_data = {'latitude': [lat], 'longitude': [lon]}
                st.map(map_data)
            else:
                st.write("No suggestions found. Please refine your search.")
    except requests.exceptions.ConnectTimeout:
        st.error("⏱️ Connection to geocoding service timed out. Please try again or enter coordinates manually below.")
    except requests.exceptions.ConnectionError:
        st.error("🌐 Unable to connect to geocoding service. Please check your internet connection or enter coordinates manually.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching location suggestions: {str(e)}")
        st.info("💡 You can still make predictions by entering latitude and longitude manually below.")

with st.form(key='rent_form'):
    
    st.subheader('Property Details', anchor=False)
    
    col1, col2 = st.columns(2)
    with col1:
        rooms = st.number_input('Rooms', min_value=1, key='rooms')
        area = np.log1p(st.number_input('Area m^2', min_value=30, key='area'))
    
    with col2:
        bathrooms = st.number_input('Bathrooms', min_value=1, key='bathrooms')  
        selected_energy_class = st.selectbox('Energy Class',
                                             options=ENERGY_CLASSES,
                                             key='energy_class',
                                             index=len(ENERGY_CLASSES) // 2)
        
    st.divider()

    col3, col4, col5 = st.columns(3)
    with col3:
        st.caption("Amenities")
        parking_spots = st.radio('Parking spots', options=['No', 'Yes'], key='parking spots', horizontal=True)
        balcony = st.radio('Balcony', options=['No', 'Yes'], key='balcony', horizontal=True)
        shared_garden = st.radio('Shared Garden', options=['No', 'Yes'], key='shared garden', horizontal=True)
        cellar = st.radio('Cellar', options=['No', 'Yes'], key='Celler', horizontal=True)

    with col4:
        st.caption("Features")
        furnished = st.radio('Furnished', options=['No', 'Yes'], key='furnished', horizontal=True)
        fiber_optic = st.radio('Fiber Optic', options=['No', 'Yes'], key='fiber optic', horizontal=True)
        electric_gate = st.radio('Electric Gate', options=['No', 'Yes'], key='electric gate', horizontal=True)
        central_heating = st.radio('Central Heating', options=['No', 'Yes'], key='Central Heating', horizontal=True)

    with col5:
        st.caption("Condition & Layout")
        condition = st.radio('Condition', options=['Good/Habitable', 'Excellent/Renovated'], key='Condition', horizontal=True)
        top_floor = st.radio('Top Floor', options=['No', 'Yes'], key='Top Floor', horizontal=True)
        external_exposure = st.radio('External Exposure', options=['No', 'Yes'], key='external exposure', horizontal=True)
    
    submit_button = st.form_submit_button(label='Predict Rent', type="primary")

if submit_button:
    if 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.error("Please search for and select a valid address first.")
    else:
        lat = st.session_state['latitude']
        lon = st.session_state['longitude']
    
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

    condition_feature = 1 if condition == 'Excellent/Renovated' else 0
    energy_class_feature = ENERGY_CLASS_MAP[selected_energy_class]
    furnished_and_central_heating = 1 if furnished == 1 and central_heating == 1 else 0
    building_layout = calculate_building_layout(cellar, top_floor)
    
    features = np.array([[
        parking_spots, bathrooms, rooms, energy_class_feature, 
        central_heating, area, furnished, balcony, 
        external_exposure, fiber_optic, electric_gate, shared_garden, 
        building_layout, furnished_and_central_heating] + [lat, lon, lat, lon, lat, lon] + [condition_feature]])
    
    # Progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(0)
    for i in range(101):
        time.sleep(0.01)
        
        progress_bar.progress(i)
    
    # Predict rent price with confidence and feature contributions
    result = predict_rent(features)
    
    if result[0] is not None:
        euro_est, lower_bound, upper_bound, confidence_score, top_contributors = result
        
        # Display main prediction
        st.success("Prediction successful!")
        st.header(f'Predicted Rent Price: €{euro_est:.0f}', anchor=False)
        
        # Display confidence score with visual indicator
        confidence_level = format_confidence_level(confidence_score)
        col_conf1, col_conf2 = st.columns([2, 1])
        
        with col_conf1:
            st.subheader(f"Confidence: {confidence_level}", anchor=False)
            st.progress(confidence_score / 100)
        
        with col_conf2:
            st.metric("Confidence Score", f"{confidence_score:.1f}%")
        
        # Display prediction range
        st.subheader(f"95% Confidence Interval: €{lower_bound:.0f} - €{upper_bound:.0f}", anchor=False)
        range_width = upper_bound - lower_bound
        st.caption(f"Range width: €{range_width:.0f} (±{(range_width / euro_est * 100 / 2):.1f}%)")
        
        st.divider()
        
        # Display feature importance explanation
        st.subheader("🎯 What's Driving This Price?", anchor=False)
        st.caption("How each feature changes the rent compared to a baseline property:")
        
        for i, contrib in enumerate(top_contributors, 1):
            contribution_text = format_contribution_text(contrib)
            impact = contrib['contribution_euro']
            
            # Use different emoji based on impact
            if abs(impact) > 100:
                emoji = "🔥" if impact > 0 else "❄️"
            elif abs(impact) > 50:
                emoji = "⭐" if impact > 0 else "⬇️"
            else:
                emoji = "📊"
            
            # Display with visual indicator - single row
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{emoji} **{i}.** {contribution_text}")
            with col2:
                amount_color = "🟢" if impact > 0 else "🔴"
                st.write(f"{amount_color} **€{abs(impact):.0f}**")
            with col3:
                direction = "↑" if impact > 0 else "↓"
                st.write(f"{direction} {abs(impact/euro_est*100):.1f}%")
        
        st.caption(f"📍 Based on location: {st.session_state.get('address', 'Unknown')}")
        
        # Add explanation tooltip
        with st.expander("ℹ️ How to interpret these results"):
            st.markdown("""
            **Confidence Score:** Indicates how certain the model is about this prediction. 
            - Higher score = more similar properties in training data
            - Lower score = fewer comparable properties, prediction more uncertain
            
            **Confidence Interval:** The range where the actual rent is likely to fall (95% probability).
            
            **Feature Contributions:** Shows how each feature affects YOUR price vs a baseline property.
            - **"Location adds €200"** means your location increases rent by €200 compared to an average location
            - **"Area reduces €50"** means your property size decreases rent by €50 compared to average
            - These are marginal contributions - they show the impact of each specific feature
            - They don't add up to the total price (that's normal for non-linear models)
            """)
    else:
        st.error("Prediction failed. Please check your inputs and try again.")

