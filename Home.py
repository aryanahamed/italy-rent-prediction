import streamlit as st
import joblib
import numpy as np

model = joblib.load('rent_prediction_model/rent_model.pkl')

st.title('Predict Rent Prices in Italy 🇮🇹')

values = {
    'parking spots': None,
    'bathrooms': None,
    'rooms': None,
    'energy class': None,
    'central heating': None,
    'area': None,
    'furnished': None,
    'balcony': None,
    'external exposure': None,
    'fiber optic': None,
    'electric gate': None,
    'shared garden': None,
    'Building Layout': None,
    'region': None,
    'city': None,
    'condition': None
}



region_city_map = {
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

regions = list(region_city_map.keys())
energy_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


def update_city_options():
    st.session_state.city = region_city_map[st.session_state.region][0]


if 'region' not in st.session_state:
    st.session_state.region = regions[0]
    st.session_state.city = region_city_map[regions[0]][0]

selected_region = st.selectbox('Region', options=regions, key='region', on_change=update_city_options)
selected_city = st.selectbox('City', options=region_city_map[st.session_state.region], key='city')

with st.form(key='rent_form'):
    rooms = st.number_input('Rooms', min_value=1, key='rooms')
    bathrooms = st.number_input('Bathrooms', min_value=1, key='bathrooms')  
    area = np.log1p(st.number_input('Area m^2', min_value=1, key='area'))
    selected_energy_class = st.selectbox('Energy Class', options=energy_classes, key='energy_class')

    col1, col2 = st.columns(2)
    with col1:
        parking_spots = st.radio('Parking spots', options=['Yes', 'No'], key='parking spots')
        balcony = st.radio('Balcony', options=['Yes', 'No'], key='balcony')
        fiber_optic = st.radio('Fiber Optic', options=['Yes', 'No'], key='fiber optic')
        shared_garden = st.radio('Shared Garden', options=['Yes', 'No'], key='shared garden')
        cellar = st.radio('Celler', options=['Yes', 'No'], key='Celler')
        central_heating = st.radio('Central Heating', options=['Yes', 'No'], key='Central Heating')

    with col2:
        furnished = st.radio('Furnished', options=['Yes', 'No'], key='furnished')
        external_exposure = st.radio('External Exposure', options=['Yes', 'No'], key='external exposure')
        electric_gate = st.radio('Electric Gate', options=['Yes', 'No'], key='electric gate')
        top_floor = st.radio('Top Floor', options=['Yes', 'No'], key='Top Floor')
        condition = st.radio('Condition', options=['Good/Habitable', 'Excellent/Renovated'], key='Condition')
    
    submit_button = st.form_submit_button(label='Submit')
    

def encode_categorical(selected_region, selected_city):
    region_columns = ['Emilia-Romagna', 'Lazio', 'Lombardy', 'Piemonte']
    city_columns = ['Genova', 'Milano', 'Other', 'Roma', 'Torino']
    
    region_encoded = [1 if col == selected_region else 0 for col in region_columns]
    city_encoded = [1 if col == selected_city else 0 for col in city_columns]
    
    return region_encoded + city_encoded

parking_spots = 1 if parking_spots == 'Yes' else 0
balcony = 1 if balcony == 'Yes' else 0
fiber_optic = 1 if fiber_optic == 'Yes' else 0
shared_garden = 1 if shared_garden == 'Yes' else 0
cellar = 1 if cellar == 'Yes' else 0
furnished = 1 if furnished == 'Yes' else 0
external_exposure = 1 if external_exposure == 'Yes' else 0
electric_gate = 1 if electric_gate == 'Yes' else 0
top_floor = 1 if top_floor == 'Yes' else 0
central_heating = 1 if central_heating == 'Yes' else 0

condition_feature = 1 if condition == 'Excellent/Renovated' else 0
energy_class_feature = 1 if selected_energy_class in ['A', 'B', 'C', 'D', 'E'] else 0
furnished_and_central_heating = 1 if furnished and central_heating == 'Yes' else 0
encoded_features = encode_categorical(selected_region, selected_city)

building_layout = 0
if cellar and top_floor:
    building_layout = 3
elif cellar:
    building_layout = 1
elif top_floor:
    building_layout = 2
    
    
if submit_button:
    
    features = np.array([[
    parking_spots, bathrooms, rooms, energy_class_feature, 
    central_heating, area, furnished, balcony, 
    external_exposure, fiber_optic, electric_gate, shared_garden, 
    building_layout, furnished_and_central_heating] + encoded_features + [condition_feature]])
    prediction = model.predict(features)
    st.header(f'Predicted Rent Price: :blue[_€{np.expm1(prediction[0]):.2f}_]', anchor=None)
    
    
    margin_of_error = 0.18176699637336002
    
    lower_bound_log = (prediction[0]) - margin_of_error
    upper_bound_log = (prediction[0]) + margin_of_error
    
    y_pred_original = np.expm1(prediction[0])
    lower_bound_original = np.expm1(lower_bound_log)
    upper_bound_original = np.expm1(upper_bound_log)
    
    st.header(f"Predicted Range: :blue[_€{lower_bound_original:.2f}_ - _€{upper_bound_original:.2f}_]", anchor=None)

