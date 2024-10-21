import time
import streamlit as st
import joblib
import numpy as np

model = joblib.load('rent_prediction_model/rent_model.pkl')

MARGIN_OF_ERROR = 0.18176699637336002
REGION_CITY_MAP = {
    'Lombardia': ['Milano', 'Bergamo', 'Brescia', 'Como', 'Cremona'],
    'Lazio': ['Roma', 'Frosinone', 'Latina', 'Rieti', 'Viterbo'],
    'Campania': ['Napoli', 'Avellino', 'Benevento', 'Caserta', 'Salerno'],
    'Sicily': ['Palermo', 'Catania', 'Messina', 'Syracuse', 'Trapani'],
    'Veneto': ['Venice', 'Verona', 'Padua', 'Vicenza', 'Treviso'],
    'Emilia-Romagna': ['Bologna', 'Parma', 'Modena', 'Ravenna', 'Reggio Emilia'],
    'Piemonte': ['Torino', 'Alessandria', 'Asti', 'Biella', 'Cuneo'],
    'Tuscany': ['Firenze', 'Pisa', 'Siena', 'Arezzo', 'Lucca'],
    'Apulia': ['Bari', 'Brindisi', 'Foggia', 'Lecce', 'Taranto'],
    'Calabria': ['Catanzaro', 'Cosenza', 'Reggio Calabria', 'Crotone', 'Vibo Valentia'],
    'Other': ['Other']
}
REGIONS = list(REGION_CITY_MAP.keys())
ENERGY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

ENERGY_CLASS_MAP = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 2,
    'E': 1,
    'F': 0,
    'G': 0
}

st.title('Predict Rent Prices in Italy 🇮🇹', anchor=False)

def update_city_options():
    """Update city options based on selected region."""
    st.session_state.city = REGION_CITY_MAP[st.session_state.region][0]

def encode_categorical(selected_region, selected_city):
    """Encode categorical features for region and city."""
    region_columns = ['Emilia-Romagna', 'Lazio', 'Lombardy', 'Piemonte']
    city_columns = ['Genova', 'Milano', 'Other', 'Roma', 'Torino']
    
    region_encoded = [1 if col == selected_region else 0 for col in region_columns]
    city_encoded = [1 if col == selected_city else 0 for col in city_columns]
    
    return region_encoded + city_encoded

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
    prediction = model.predict(features)
    log_estimate = prediction[0]
    euro_est = np.expm1(log_estimate)
    
    lower_bound_log = log_estimate - MARGIN_OF_ERROR
    upper_bound_log = log_estimate + MARGIN_OF_ERROR
    
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    
    return euro_est, lower_bound, upper_bound

# Initialize session state
if 'region' not in st.session_state:
    st.session_state.region = REGIONS[0]
    st.session_state.city = REGION_CITY_MAP[REGIONS[0]][0]
    
col1, col2 = st.columns(2)
with col1:
    selected_region = st.selectbox('Region', options=REGIONS, key='region', on_change=update_city_options)
with col2:
    selected_city = st.selectbox('City', options=REGION_CITY_MAP[st.session_state.region], key='city')

with st.form(key='rent_form'):
    col3, col4 = st.columns(2)
    with col3:
        rooms = st.number_input('Rooms', min_value=1, key='rooms')
        area = np.log1p(st.number_input('Area m^2', min_value=30, key='area'))
    
    with col4:
        bathrooms = st.number_input('Bathrooms', min_value=1, key='bathrooms')  
        selected_energy_class = st.selectbox('Energy Class', options=ENERGY_CLASSES, key='energy_class')

    col5, col6, col7 = st.columns(3)
    with col5:
        parking_spots = st.radio('Parking spots', options=['Yes', 'No'], key='parking spots')
        balcony = st.radio('Balcony', options=['Yes', 'No'], key='balcony')
        fiber_optic = st.radio('Fiber Optic', options=['Yes', 'No'], key='fiber optic')
        shared_garden = st.radio('Shared Garden', options=['Yes', 'No'], key='shared garden')

    with col6:
        furnished = st.radio('Furnished', options=['Yes', 'No'], key='furnished')
        external_exposure = st.radio('External Exposure', options=['Yes', 'No'], key='external exposure')
        electric_gate = st.radio('Electric Gate', options=['Yes', 'No'], key='electric gate')
        top_floor = st.radio('Top Floor', options=['Yes', 'No'], key='Top Floor')
        
    
    with col7:
        cellar = st.radio('Celler', options=['Yes', 'No'], key='Celler')
        condition = st.radio('Condition', options=['Good/Habitable', 'Excellent/Renovated'], key='Condition')
        central_heating = st.radio('Central Heating', options=['Yes', 'No'], key='Central Heating')
    
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
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
    print(selected_energy_class, energy_class_feature)
    furnished_and_central_heating = 1 if furnished and central_heating == 'Yes' else 0
    encoded_features = encode_categorical(selected_region, selected_city)
    building_layout = calculate_building_layout(cellar, top_floor)
    
    features = np.array([[
        parking_spots, bathrooms, rooms, energy_class_feature, 
        central_heating, area, furnished, balcony, 
        external_exposure, fiber_optic, electric_gate, shared_garden, 
        building_layout, furnished_and_central_heating] + encoded_features + [condition_feature]])
    
    # Add progress bar
    progress_bar = st.progress(0)
    progress_bar.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i)
    
    # Predict rent price
    euro_est, lower_bound, upper_bound = predict_rent(features)
    
    # Display results
    st.header(f'Predicted Rent Price: :blue[€{euro_est:.2f}]', anchor=False)
    st.subheader(f"Predicted Range: :blue[€{lower_bound:.2f} - €{upper_bound:.2f}]", anchor=False)

