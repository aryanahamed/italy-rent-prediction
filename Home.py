import time
import streamlit as st
import joblib
import numpy as np
import requests
from config import ENERGY_CLASSES, ENERGY_CLASS_MAP, MARGIN_OF_ERROR


@st.cache_resource(ttl=3600)
def load_model(path='rent_prediction_model/rent_model_v2.pkl'):
    """Loads the pre-trained model from the specified path."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None
    
model = load_model()

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
    """Predict rent price and calculate prediction range."""
    if model is None:
        st.error("Model is not loaded. Cannot perform prediction.")
        return None, None, None
    
    prediction = model.predict(features)
    log_estimate = prediction[0]
    euro_est = np.expm1(log_estimate)
    
    lower_bound_log = log_estimate - MARGIN_OF_ERROR
    upper_bound_log = log_estimate + MARGIN_OF_ERROR
    
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    
    lower_bound = max(0, lower_bound)
    upper_bound = max(lower_bound, upper_bound)
    
    return euro_est, lower_bound, upper_bound

    

address = st.text_input(
    'Search your desired neighborhood (For better results add city with the name)', 
    key='address',
    help="Example: 'Navigli, Milano'"
    )

if address:
    photon_url = f"https://photon.komoot.io/api/?q={address} Italy&limit=5"
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
    else:
        st.write("Error fetching location suggestions.")

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
    
    # Predict rent price
    euro_est, lower_bound, upper_bound = predict_rent(features)
    
    # Display results
    if euro_est is not None:
        st.success("Prediction successful!")
        st.header(f'Predicted Rent Price: €{euro_est:.0f}', anchor=False)
        st.subheader(f"Estimated Range: €{lower_bound:.0f} - €{upper_bound:.0f}", anchor=False)
        st.caption(f"Based on location: {st.session_state['address']}")

