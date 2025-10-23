import time
import streamlit as st
import joblib
import numpy as np
import requests
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
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




