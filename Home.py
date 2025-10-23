import time
import streamlit as st
import joblib
import numpy as np
import requests
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from config import ENERGY_CLASSES, ENERGY_CLASS_MAP, MARGIN_OF_ERROR
from prediction_utils import (
    PredictionAnalyzer,
    format_confidence_level,
    format_contribution_text
)
from map_data import (
    load_neighborhood_price_data, 
    get_italy_center_coords,
    load_property_cluster_data,
    get_price_category
)
from feature_utils import (
    find_similar_properties,
    get_historical_price_trends,
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
    st.stop()  # Stop execution if model is not available

# Initialize session state for predictions
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

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
    """
    Make rent prediction with comprehensive error handling.
    
    Args:
        features: Feature array for prediction
        
    Returns:
        Tuple of (euro_est, lower_bound, upper_bound, confidence_score, top_contributors)
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
        
        # Get confidence-based prediction with intervals
        log_estimate, log_lower, log_upper, confidence_score = analyzer.calculate_confidence_score(features)
        
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
        
        return euro_est, lower_bound, upper_bound, confidence_score, top_contributors
    
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
        Tuple of (region, city, neighborhood, state) as strings
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
        url = f"https://photon.komoot.io/api/?q={location_name}, {country}&limit=1"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            features = response.json().get('features', [])
            if features:
                coords = features[0]['geometry']['coordinates']
                return coords[1], coords[0]  # lat, lon
    except:
        pass
    
    return None, None

address = st.text_input(
    'Search your desired neighborhood (For better results add city with the name)', 
    key='address',
    help="Example: 'Navigli, Milano'"
    )

# Issue #5 fix: Empty address validation
if address and address.strip():
    photon_url = f"https://photon.komoot.io/api/?q={address} Italy&limit=5"
    try:
        response = requests.get(photon_url, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 200:
            suggestions = response.json()['features']
            
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
                
                selected_option = st.selectbox("Select an option", options)
                
                # Get the correct place using the mapping
                selected_place = suggestions[place_mapping[selected_option]]
                lat = selected_place['geometry']['coordinates'][1]
                lon = selected_place['geometry']['coordinates'][0]
                
                # Extract location hierarchy
                properties = selected_place['properties']
                state, city, neighborhood = extract_location_hierarchy(properties)
                
                # Store primary location coordinates
                st.session_state['latitude'] = lat
                st.session_state['longitude'] = lon
                
                # Geocode region level (state)
                if state:
                    region_lat, region_lon = geocode_location_level(state)
                    st.session_state['latitude_region'] = region_lat if region_lat else lat
                    st.session_state['longitude_region'] = region_lon if region_lon else lon
                    st.session_state['region'] = state
                else:
                    st.session_state['latitude_region'] = lat
                    st.session_state['longitude_region'] = lon
                    st.session_state['region'] = ''
                
                # Geocode city level
                if city:
                    city_lat, city_lon = geocode_location_level(city)
                    st.session_state['latitude_city'] = city_lat if city_lat else lat
                    st.session_state['longitude_city'] = city_lon if city_lon else lon
                    st.session_state['city'] = city
                else:
                    st.session_state['latitude_city'] = lat
                    st.session_state['longitude_city'] = lon
                    st.session_state['city'] = ''
                
                # Neighborhood level uses primary coordinates
                st.session_state['latitude_neighborhood'] = lat
                st.session_state['longitude_neighborhood'] = lon
                st.session_state['neighborhood'] = neighborhood if neighborhood else properties.get('name', '')
                
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
    # Issue #5 fix: Improved validation with clear messaging
    if not address or not address.strip():
        st.error("⚠️ Please enter an address in the search field above before making a prediction.")
    elif 'latitude' not in st.session_state or 'longitude' not in st.session_state:
        st.error("⚠️ Please select a valid address from the dropdown suggestions before making a prediction.")
    else:
        # Store area in a separate session state key (can't use 'area' as it's bound to widget)
        st.session_state['area_log_value'] = area
        
        # Get all location coordinates (region, city, neighborhood)
        lat = st.session_state['latitude']
        lon = st.session_state['longitude']
        lat_region = st.session_state.get('latitude_region', lat)
        lon_region = st.session_state.get('longitude_region', lon)
        lat_city = st.session_state.get('latitude_city', lat)
        lon_city = st.session_state.get('longitude_city', lon)
        lat_neighborhood = st.session_state.get('latitude_neighborhood', lat)
        lon_neighborhood = st.session_state.get('longitude_neighborhood', lon)
    
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
    
    # Build feature array with separate region/city/neighborhood coordinates
    features = np.array([[
        parking_spots, bathrooms, rooms, energy_class_feature, 
        central_heating, area, furnished, balcony, 
        external_exposure, fiber_optic, electric_gate, shared_garden, 
        building_layout, furnished_and_central_heating,
        lat_region, lon_region,  # Region-level coordinates
        lat_city, lon_city,      # City-level coordinates
        lat_neighborhood, lon_neighborhood,  # Neighborhood-level coordinates
        condition_feature
    ]])
    
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
        
        # Store results in session state
        st.session_state.prediction_results = {
            'euro_est': euro_est,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_score': confidence_score,
            'top_contributors': top_contributors,
            'address': st.session_state.get('address', 'Unknown')
        }

# Display prediction results if they exist in session state
if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    euro_est = results['euro_est']
    lower_bound = results['lower_bound']
    upper_bound = results['upper_bound']
    confidence_score = results['confidence_score']
    top_contributors = results['top_contributors']
    
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
    
    st.caption(f"📍 Based on location: {results['address']}")
    
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
    
    st.divider()
    
    # ========== FEATURE #1: AFFORDABILITY CALCULATOR ==========
    st.subheader("💰 Affordability Analysis", anchor=False)
    
    # Issue #7 fix: Extract region from session state for regional salary data
    region_name = st.session_state.get('region', None)
    city_name = st.session_state.get('city', None)
    
    # Display location context if available
    if region_name:
        st.caption(f"📍 Region: {region_name}" + (f" | City: {city_name}" if city_name else ""))
    
    affordability = calculate_affordability(euro_est, region=region_name)
    
    # Add explanation at the top
    st.info(f"""
    **What does this mean?** This analysis shows whether the €{euro_est:.0f}/month rent is affordable based on the **30% rule**: 
    rent should not exceed 30% of monthly income. The average {"regional" if region_name else "Italian"} salary is €{affordability['avg_salary']:.0f}/month.
    """)
    
    col_aff1, col_aff2, col_aff3 = st.columns(3)
    
    with col_aff1:
        st.metric(
            "Affordability Level",
            affordability['affordability_level'],
            delta=None
        )
        st.caption(f"{affordability['affordability_emoji']} Overall assessment")
    
    with col_aff2:
        st.metric(
            "Required Monthly Income",
            f"€{affordability['required_income']:.0f}",
            delta=None
        )
        st.caption("To afford this rent comfortably (30% rule)")
    
    with col_aff3:
        salary_label = "Regional Salary" if region_name else "Avg Italian Salary"
        st.metric(
            f"Rent as % of {salary_label}",
            f"{affordability['pct_of_avg_salary']:.1f}%",
            delta=f"{affordability['pct_of_avg_salary'] - 30:.1f}% vs 30% threshold",
            delta_color="inverse"
        )
        st.caption(f"Reference: €{affordability['avg_salary']:.0f}/mo average salary")
    
    # Affordability progress bar with clearer label
    st.write("**Rent Burden (% of average salary):**")
    affordability_pct = min(affordability['pct_of_avg_salary'], 100)
    st.progress(affordability_pct / 100)
    col_legend1, col_legend2 = st.columns([1, 3])
    with col_legend1:
        st.caption(f"**Current: {affordability['pct_of_avg_salary']:.1f}%**")
    with col_legend2:
        st.caption("✅ ≤30% Affordable | 🟡 30-40% Moderate | 🟠 40-50% Challenging | 🔴 >50% High burden")
    
    if affordability['is_affordable']:
        st.success("✅ This rent is within the affordable range (≤30% of average income)")
    else:
        st.warning(f"⚠️ This rent exceeds the recommended 30% affordability threshold by {affordability['pct_of_avg_salary'] - 30:.1f} percentage points")
    
    st.divider()
    
    # ========== FEATURE #3: SIMILAR PROPERTIES ==========
    st.subheader("🏘️ Similar Properties in the Area", anchor=False)
    st.caption("Actual listings with similar characteristics from our database")
    
    with st.spinner("Finding similar properties..."):
        # Get input values from session state if they exist
        input_rooms = st.session_state.get('rooms', 3)
        input_area = st.session_state.get('area', np.log1p(80))  # Default 80 sqm
        
        similar_props = find_similar_properties(
            city=city_name if city_name else "Milano",
            rooms=input_rooms,
            area=input_area,
            price=euro_est,
            top_n=5
        )
    
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
            diff = euro_est - avg_similar_price
            st.metric("Similar Props Avg", f"€{avg_similar_price:.0f}", 
                     delta=f"{diff:+.0f}" if abs(diff) > 1 else "Similar")
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
                    price_diff = euro_est - prop['price']
                    if abs(price_diff) < 50:
                        st.write(f"💚 Very similar: €{prop['price']:.0f}")
                    elif price_diff > 0:
                        st.write(f"🔵 Cheaper: -€{abs(price_diff):.0f}")
                    else:
                        st.write(f"🔴 Pricier: +€{abs(price_diff):.0f}")
                    st.write(f"• Match: {prop['similarity_score']*100:.0f}%")
    else:
        st.info("No similar properties found in our database for this search criteria.")
    
    st.divider()
    
    # ========== FEATURE #2: HISTORICAL PRICE TRENDS ==========
    st.subheader("📈 Historical Price Trends", anchor=False)
    st.caption("Price evolution in this area over time")
    
    # Issue #8 fix: Add neighborhood-level filtering option
    neighborhood_name = st.session_state.get('neighborhood', None)
    
    col_filter1, col_filter2 = st.columns([3, 1])
    with col_filter1:
        st.write("**Filter trends by:**")
    with col_filter2:
        show_neighborhood = st.checkbox(
            "Neighborhood only", 
            value=False, 
            key='filter_neighborhood',
            help="Show trends for specific neighborhood instead of city-wide data"
        ) if neighborhood_name else False
    
    with st.spinner("Loading historical data..."):
        if show_neighborhood and neighborhood_name:
            monthly_data, hist_stats = get_historical_price_trends(
                city=city_name if city_name else "Milano",
                neighborhood=neighborhood_name,
                min_samples=5
            )
            location_label = f"{neighborhood_name}, {city_name}" if city_name else neighborhood_name
        else:
            monthly_data, hist_stats = get_historical_price_trends(
                city=city_name if city_name else "Milano",
                neighborhood=None,
                min_samples=5
            )
            location_label = city_name if city_name else "Milano"
    
    if monthly_data is not None and not monthly_data.empty:
        # Display statistics
        col_hist1, col_hist2, col_hist3, col_hist4 = st.columns(4)
        
        with col_hist1:
            # Use actual price change for trend direction (more intuitive)
            if 'price_change_pct' in hist_stats:
                price_change = hist_stats['price_change_pct']
                if price_change > 0:
                    trend_emoji = "📈"
                    trend_label = "Rising"
                else:
                    trend_emoji = "📉"
                    trend_label = "Falling"
                st.metric("Market Trend", f"{trend_emoji} {trend_label}", 
                         delta=f"{price_change:+.1f}%")
        
        with col_hist2:
            if 'data_points' in hist_stats:
                st.metric("Data Points", f"{hist_stats['data_points']} months")
        
        with col_hist3:
            st.metric("Avg Price in Area", f"€{hist_stats['avg_price_overall']:.0f}")
        
        with col_hist4:
            st.metric("Total Listings", f"{hist_stats['total_listings']}")
        
        # Create time series chart using Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['avg_price'],
            mode='lines+markers',
            name='Average Price',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x|%B %Y}</b><br>Avg Price: €%{y:.0f}<extra></extra>'
        ))
        
        # Add your prediction as a horizontal line
        fig.add_hline(
            y=euro_est, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Your Prediction: €{euro_est:.0f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f"Monthly Average Rent Prices Over Time - {location_label}",
            xaxis_title="Date",
            yaxis_title="Average Rent (€/month)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation with location context - use actual price change direction
        location_context = f" in {location_label}"
        if 'price_change_pct' in hist_stats:
            price_change_pct = hist_stats.get('price_change_pct', 0)
            if price_change_pct > 0:
                st.info(f"📊 The market{location_context} is trending upward. Prices have increased by {abs(price_change_pct):.1f}% over the analyzed period.")
            else:
                st.info(f"📊 The market{location_context} is trending downward. Prices have decreased by {abs(price_change_pct):.1f}% over the analyzed period.")
    else:
        st.info("Insufficient historical data available for this location. Try a more popular city or neighborhood.")
    
    st.divider()
    
    # ========== FEATURE #4: PROPERTY FEATURE OPTIMIZER ==========
    st.subheader("🎨 Property Feature Optimizer - 'What If' Scenarios", anchor=False)
    st.caption("Explore how different features affect the rent price in real-time")
    
    with st.expander("🔧 Try Different Feature Combinations", expanded=False):
        st.write("**Adjust features below to see how they impact the predicted rent:**")
        
        # Get current values from session state
        current_parking = st.session_state.get('parking spots', 'No')
        current_balcony = st.session_state.get('balcony', 'No')
        current_furnished = st.session_state.get('furnished', 'No')
        current_fiber = st.session_state.get('fiber optic', 'No')
        current_heating = st.session_state.get('Central Heating', 'No')
        current_garden = st.session_state.get('shared garden', 'No')
        current_gate = st.session_state.get('electric gate', 'No')
        
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
            opt_rooms_adj = st.slider("Adjust Rooms", min_value=1, max_value=10, 
                                       value=st.session_state.get('rooms', 3), key='opt_rooms')
        
        # Additional adjustable features (Issue #4 fix)
        st.write("**Adjust Core Features:**")
        col_adj1, col_adj2, col_adj3 = st.columns(3)
        
        with col_adj1:
            opt_bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, 
                                      value=int(st.session_state.get('bathrooms', 1)), key='opt_bathrooms')
        
        with col_adj2:
            # Get area from stored log value or widget state, with fallback
            current_area_log = st.session_state.get('area_log_value', None)
            if current_area_log is None:
                # Try to get from widget state
                area_sqm_widget = st.session_state.get('area', 80)
                current_area_log = np.log1p(area_sqm_widget)
            
            # Safety check: ensure area value is within reasonable bounds
            try:
                current_area_sqm = int(np.expm1(current_area_log))
                # Clamp to valid range
                current_area_sqm = max(30, min(300, current_area_sqm))
            except (ValueError, OverflowError):
                current_area_sqm = 80  # Default fallback
            
            opt_area_sqm = st.slider("Area (m²)", min_value=30, max_value=300, 
                                     value=current_area_sqm, key='opt_area_sqm')
            opt_area = np.log1p(opt_area_sqm)
        
        with col_adj3:
            current_energy_idx = ENERGY_CLASSES.index(st.session_state.get('energy_class', 'D'))
            opt_energy_class = st.selectbox("Energy Class", options=ENERGY_CLASSES, 
                                            index=current_energy_idx, key='opt_energy_class')
        
        # Calculate new prediction with adjusted features
        if st.button("🔄 Calculate New Price", type="primary", key='optimize_btn'):
            # Get all current values
            opt_external = st.session_state.get('external exposure', 'No')
            opt_cellar = st.session_state.get('Celler', 'No')
            opt_top_floor = st.session_state.get('Top Floor', 'No')
            opt_condition = st.session_state.get('Condition', 'Good/Habitable')
            
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
            
            # Calculate derived features
            new_condition_feature = 1 if opt_condition == 'Excellent/Renovated' else 0
            new_energy_feature = ENERGY_CLASS_MAP[opt_energy_class]
            new_furnished_heating = 1 if new_furnished == 1 and new_heating == 1 else 0
            new_building_layout = calculate_building_layout(new_cellar, new_top_floor)
            
            # Get location coordinates (all three levels)
            opt_lat = st.session_state.get('latitude', 45.4642)
            opt_lon = st.session_state.get('longitude', 9.1900)
            opt_lat_region = st.session_state.get('latitude_region', opt_lat)
            opt_lon_region = st.session_state.get('longitude_region', opt_lon)
            opt_lat_city = st.session_state.get('latitude_city', opt_lat)
            opt_lon_city = st.session_state.get('longitude_city', opt_lon)
            opt_lat_neighborhood = st.session_state.get('latitude_neighborhood', opt_lat)
            opt_lon_neighborhood = st.session_state.get('longitude_neighborhood', opt_lon)
            
            # Build feature array with proper location hierarchy
            new_features = np.array([[
                new_parking, opt_bathrooms, opt_rooms_adj, new_energy_feature,
                new_heating, opt_area, new_furnished, new_balcony,
                new_external, new_fiber, new_gate, new_garden,
                new_building_layout, new_furnished_heating,
                opt_lat_region, opt_lon_region,      # Region-level
                opt_lat_city, opt_lon_city,          # City-level
                opt_lat_neighborhood, opt_lon_neighborhood,  # Neighborhood-level
                new_condition_feature
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
                
                # Show feature impact breakdown
                st.write("**Impact Breakdown:**")
                impacts = []
                if opt_parking != (current_parking == 'Yes'):
                    impacts.append(f"{'Added' if opt_parking else 'Removed'} Parking")
                if opt_balcony != (current_balcony == 'Yes'):
                    impacts.append(f"{'Added' if opt_balcony else 'Removed'} Balcony")
                if opt_furnished != (current_furnished == 'Yes'):
                    impacts.append(f"{'Added' if opt_furnished else 'Removed'} Furnished")
                if opt_heating != (current_heating == 'Yes'):
                    impacts.append(f"{'Added' if opt_heating else 'Removed'} Central Heating")
                if opt_rooms_adj != st.session_state.get('rooms', 3):
                    impacts.append(f"Changed rooms: {st.session_state.get('rooms', 3)} → {opt_rooms_adj}")
                
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
    report_text = generate_prediction_report(
        prediction_data=results,
        similar_properties=similar_props if similar_props else [],
        affordability=affordability,
        historical_stats=hist_stats if hist_stats else None
    )
    
    # Download button
    col_dl1, col_dl2 = st.columns([2, 1])
    with col_dl1:
        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=report_text,
            file_name=f"rent_prediction_report_{results['address'].replace(' ', '_').replace(',', '')}_{euro_est:.0f}EUR.txt",
            mime="text/plain",
            help="Download a comprehensive text report with all prediction details"
        )
    
    with col_dl2:
        st.caption(f"Report size: {len(report_text)} chars")


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
            center_lat, center_lon = get_italy_center_coords()
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
                import pandas as pd
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
        center_lat, center_lon = get_italy_center_coords()
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
                **{color_emoji[color]} {label}**  
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




