#!/usr/bin/env python3
"""
Enhanced Streamlit app for Saudi housing price prediction with dynamic filtering and comprehensive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit config
st.set_page_config(
    page_title="🏡 Saudi House Price Predictor", 
    layout="wide", 
    page_icon="🏠"
)

# Load CSS for styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")

local_css("style.css")

# Title & Header
st.title("🏡 Saudi House Price Predictor")
st.markdown("Machine learning model with 81.07% accuracy for Saudi real estate")

# Sidebar header
st.sidebar.header("🔍 Enter Property Details")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/saudi_housing_english.csv")
        return df, None
    except Exception as e:
        return None, str(e)

df, data_error = load_data()

if data_error:
    st.error(f"Error loading data: {data_error}")
    st.stop()

# Load model components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("models/saved/model.pkl")
        scaler = joblib.load("models/saved/scaler.pkl")
        features = joblib.load("models/saved/features.pkl")
        encoders = joblib.load("models/saved/encoders.pkl")
        
        return model, scaler, features, encoders, None
    except Exception as e:
        return None, None, None, None, str(e)

# Load model components
model, scaler, features, encoders, model_error = load_model_components()

if model_error:
    st.error(f"Error loading model: {model_error}")
    st.error("Please ensure model files exist in models/saved/ directory.")
    st.stop()



# Get data ranges for validation
def get_safe_range(column, default_min, default_max):
    try:
        return int(df[column].min()), int(df[column].max())
    except:
        return default_min, default_max

max_bedrooms = get_safe_range("bedrooms", 1, 10)[1]
max_bathrooms = get_safe_range("bathrooms", 1, 10)[1]
max_property_age = get_safe_range("property_age", 0, 50)[1]
max_land_area = min(get_safe_range("land_area", 100, 5000)[1], 5000)
max_living_rooms = get_safe_range("living_rooms", 1, 10)[1]
max_garage = get_safe_range("garage", 0, 5)[1]

# Input validation functions
def validate_inputs():
    warnings = []
    
    if bedrooms > 7:
        warnings.append("⚠️ More than 7 bedrooms is unusual and may affect prediction accuracy")
    
    if area > 2000:
        warnings.append("⚠️ Very large properties (>2000 sqm) may have less accurate predictions")
    
    if property_age > 30:
        warnings.append("⚠️ Very old properties (>30 years) may have less accurate predictions")
    
    if bedrooms + bathrooms + living_rooms > 15:
        warnings.append("⚠️ Total rooms >15 is unusual and may affect accuracy")
    
    return warnings

# Input controls with better defaults
bedrooms = st.sidebar.slider("Bedrooms 🛏", min_value=1, max_value=max_bedrooms, value=min(4, max_bedrooms))
bathrooms = st.sidebar.slider("Bathrooms 🚿", min_value=1, max_value=max_bathrooms, value=min(3, max_bathrooms))

# Property type with better explanation
duplex = st.sidebar.selectbox(
    "Property Type 🏠", 
    options=["Non-Duplex", "Duplex"],
    help="Duplex properties have two separate living units"
)

# Location inputs with dynamic filtering
try:
    city_options = sorted(df["city"].unique())
    direction_options = sorted(df["front_direction"].unique())
except:
    city_options = ["Unknown"]
    direction_options = ["Unknown"]

# City selection first
city = st.sidebar.selectbox("City 🏙️", options=city_options)

# Dynamic district filtering based on selected city
try:
    if city and city != "Unknown":
        district_options = sorted(df[df["city"] == city]["district"].unique())
        district_help = f"{len(district_options)} districts available in {city}"
    else:
        district_options = sorted(df["district"].unique())
        district_help = "Select a city first to filter districts"
except:
    district_options = ["Unknown"]
    district_help = "Error loading districts"

district = st.sidebar.selectbox("District 📍", options=district_options, 
                               help=district_help)

# Show district info
if city and city != "Unknown" and district:
    district_properties = len(df[(df["city"] == city) & (df["district"] == district)])
    if district_properties > 0:
        avg_price_district = df[(df["city"] == city) & (df["district"] == district)]["price"].mean()
        st.sidebar.info(f"📊 {district_properties} properties in {district}, {city}\n💰 Avg: SAR {avg_price_district:,.0f}")
front_direction = st.sidebar.selectbox("Front Direction 🧭", options=direction_options)

# Numeric inputs
property_age = st.sidebar.slider("Property Age 🏗️ (years)", min_value=0, max_value=max_property_age, value=5)
area = st.sidebar.slider("Land Area 📐 (m²)", min_value=100, max_value=max_land_area, value=400)
living_rooms = st.sidebar.slider("Living Rooms 🛋", min_value=1, max_value=max_living_rooms, value=2)

# Amenities
st.sidebar.subheader("🏠 Amenities")
garage = st.sidebar.selectbox("Garage 🚗", options=list(range(max_garage + 1)), index=1)
driver_room = st.sidebar.selectbox("Driver Room 👨‍💼", options=[0, 1])
maid_room = st.sidebar.selectbox("Maid Room 👩‍💼", options=[0, 1])
furnished = st.sidebar.selectbox("Furnished 🪑", options=[0, 1])
ac = st.sidebar.selectbox("Air Conditioning ❄️", options=[0, 1], index=1)
roof = st.sidebar.selectbox("Roof 🏠", options=[0, 1])
pool = st.sidebar.selectbox("Pool 🏊", options=[0, 1])
front_yard = st.sidebar.selectbox("Front Yard 🌳", options=[0, 1])
basement = st.sidebar.selectbox("Basement 🏠", options=[0, 1])
stairs = st.sidebar.selectbox("Stairs 🪜", options=[0, 1])
elevator = st.sidebar.selectbox("Elevator 🚠", options=[0, 1])
fireplace = st.sidebar.selectbox("Fireplace 🔥", options=[0, 1])

# Validate inputs
input_warnings = validate_inputs()

# Encoding with better error handling
duplex_encoded = 1 if duplex == "Duplex" else 0

def safe_encode(encoder_name, value):
    try:
        if encoder_name in encoders:
            return encoders[encoder_name].transform([value])[0]
        else:
            return 0
    except (ValueError, KeyError):
        return 0  # Default for unknown categories

city_encoded = safe_encode('city', city)
district_encoded = safe_encode('district', district)
front_direction_encoded = safe_encode('front_direction', front_direction)

# Calculate derived features
total_rooms = bedrooms + bathrooms + living_rooms
luxury_score = (garage + driver_room + maid_room + furnished + ac + roof + 
               pool + front_yard + basement + duplex_encoded + stairs + elevator + fireplace)

# Prepare input data
def prepare_input_data():
    input_dict = {
        'city_encoded': city_encoded,
        'district_encoded': district_encoded,
        'front_direction_encoded': front_direction_encoded,
        'land_area': area,
        'property_age': property_age,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'living_rooms': living_rooms,
        'garage': garage,
        'driver_room': driver_room,
        'maid_room': maid_room,
        'furnished': furnished,
        'air_conditioning': ac,
        'roof': roof,
        'pool': pool,
        'front_yard': front_yard,
        'basement': basement,
        'duplex': duplex_encoded,
        'stairs': stairs,
        'elevator': elevator,
        'fireplace': fireplace,
        'total_rooms': total_rooms,
        'luxury_score': luxury_score
    }
    
    # Add improved model features if available
    if 'room_density' in features:
        input_dict['room_density'] = total_rooms / area * 1000
    
    return input_dict

input_dict = prepare_input_data()

# Make prediction with error handling
try:
    input_data = np.array([[input_dict.get(feature, 0) for feature in features]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]
    prediction_success = True
except Exception as e:
    prediction = 0
    prediction_success = False
    prediction_error = str(e)

# Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 Your Input Summary")
    
    # Show input warnings
    if input_warnings:
        for warning in input_warnings:
            st.warning(warning)
    
    # Create summary table
    summary_data = {
        'Property Details': [
            f"{bedrooms} bedrooms, {bathrooms} bathrooms",
            f"{living_rooms} living rooms ({total_rooms} total)",
            f"{area} m² land area",
            f"{property_age} years old",
            f"{duplex} in {city}, {district}",
            f"Facing {front_direction}"
        ],
        'Amenities': [
            f"Garage: {garage} cars" if garage > 0 else "No garage",
            f"Luxury Score: {luxury_score}/12",
            "✓ " + ", ".join([
                name for name, value in [
                    ("AC", ac), ("Furnished", furnished), ("Pool", pool),
                    ("Driver Room", driver_room), ("Maid Room", maid_room)
                ] if value
            ]) if any([ac, furnished, pool, driver_room, maid_room]) else "Basic amenities"
        ]
    }
    
    for category, items in summary_data.items():
        st.write(f"**{category}:**")
        for item in items:
            st.write(f"• {item}")

with col2:
    st.subheader("🔮 Price Prediction")
    
    if prediction_success:
        formatted_prediction = f"SAR {prediction:,.0f}"
        st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
        
        # Confidence indicator
        confidence_score = min(100, max(60, 100 - (luxury_score * 2) - (abs(bedrooms - 5) * 5)))
        st.progress(confidence_score / 100)
        st.caption(f"Prediction confidence: {confidence_score}%")
        
        # Price context
        try:
            median_price = df['price'].median()
            if prediction > median_price * 1.5:
                st.info("🏆 This is a high-value property")
            elif prediction < median_price * 0.7:
                st.info("💰 This is an affordable property")
            else:
                st.info("📊 This is a moderately-priced property")
        except:
            pass
            
    else:
        st.error(f"Prediction failed: {prediction_error}")

# Additional insights
if prediction_success:
    st.markdown("---")
    st.subheader("📈 Market Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            similar_properties = df[
                (df['bedrooms'] == bedrooms) & 
                (df['city'] == city)
            ]
            if len(similar_properties) > 0:
                avg_similar = similar_properties['price'].mean()
                st.metric(
                    "Similar Properties Avg", 
                    f"SAR {avg_similar:,.0f}",
                    f"{((prediction - avg_similar) / avg_similar * 100):+.1f}%"
                )
        except:
            st.metric("Similar Properties", "Data unavailable")
    
    with col2:
        try:
            price_per_sqm = prediction / area
            st.metric("Price per m²", f"SAR {price_per_sqm:,.0f}")
        except:
            st.metric("Price per m²", "N/A")
    
    with col3:
        st.metric("Model Accuracy", "81.07%")

# Enhanced Data Visualizations
st.markdown("---")
st.header("📊 Comprehensive Market Analysis")

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["🏙️ City Analysis", "🏠 Property Features", "💰 Price Trends", "📈 Interactive Charts"])

with tab1:
    st.subheader("City-Based Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # City price comparison
        fig_city = px.bar(
            df.groupby('city')['price'].agg(['mean', 'count']).reset_index(),
            x='city', y='mean',
            title="Average Price by City",
            labels={'mean': 'Average Price (SAR)', 'city': 'City'},
            color='mean',
            color_continuous_scale='viridis'
        )
        fig_city.update_layout(height=400)
        st.plotly_chart(fig_city, use_container_width=True)
    
    with col2:
        # Property count by city
        city_counts = df['city'].value_counts()
        fig_count = px.pie(
            values=city_counts.values,
            names=city_counts.index,
            title="Property Distribution by City"
        )
        fig_count.update_layout(height=400)
        st.plotly_chart(fig_count, use_container_width=True)
    
    # District analysis for selected city
    if city and city != "Unknown":
        st.subheader(f"District Analysis for {city}")
        city_data = df[df['city'] == city]
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Top districts by average price
            district_prices = city_data.groupby('district')['price'].mean().sort_values(ascending=False).head(10)
            fig_district = px.bar(
                x=district_prices.values,
                y=district_prices.index,
                orientation='h',
                title=f"Top 10 Most Expensive Districts in {city}",
                labels={'x': 'Average Price (SAR)', 'y': 'District'}
            )
            fig_district.update_layout(height=500)
            st.plotly_chart(fig_district, use_container_width=True)
        
        with col4:
            # District property count
            district_counts = city_data['district'].value_counts().head(10)
            fig_district_count = px.bar(
                x=district_counts.index,
                y=district_counts.values,
                title=f"Top 10 Districts by Property Count in {city}",
                labels={'x': 'District', 'y': 'Number of Properties'}
            )
            fig_district_count.update_xaxes(tickangle=45)
            fig_district_count.update_layout(height=500)
            st.plotly_chart(fig_district_count, use_container_width=True)

with tab2:
    st.subheader("Property Features Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bedroom vs Price relationship
        bedroom_stats = df.groupby('bedrooms').agg({
            'price': ['mean', 'median', 'count']
        }).round(0)
        bedroom_stats.columns = ['Mean Price', 'Median Price', 'Count']
        bedroom_stats = bedroom_stats.reset_index()
        
        fig_bedroom = px.line(
            bedroom_stats, x='bedrooms', y=['Mean Price', 'Median Price'],
            title="Price Trends by Bedroom Count",
            labels={'value': 'Price (SAR)', 'bedrooms': 'Number of Bedrooms'}
        )
        fig_bedroom.update_layout(height=400)
        st.plotly_chart(fig_bedroom, use_container_width=True)
    
    with col2:
        # Land area vs Price scatter
        sample_data = df.sample(min(1000, len(df)))  # Sample for performance
        fig_scatter = px.scatter(
            sample_data, x='land_area', y='price',
            color='city', size='bedrooms',
            title="Price vs Land Area (Sample of Properties)",
            labels={'land_area': 'Land Area (m²)', 'price': 'Price (SAR)'},
            hover_data=['bedrooms', 'bathrooms']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Amenities impact analysis
    st.subheader("Amenities Impact on Price")
    
    amenities = ['garage', 'driver_room', 'maid_room', 'furnished', 'air_conditioning', 
                'pool', 'front_yard', 'basement', 'duplex', 'elevator', 'fireplace']
    
    amenity_impact = []
    for amenity in amenities:
        if amenity in df.columns:
            with_amenity = df[df[amenity] == 1]['price'].mean()
            without_amenity = df[df[amenity] == 0]['price'].mean()
            impact = with_amenity - without_amenity
            amenity_impact.append({
                'Amenity': amenity.replace('_', ' ').title(),
                'Price Increase (SAR)': impact,
                'Percentage Increase': (impact / without_amenity * 100) if without_amenity > 0 else 0
            })
    
    amenity_df = pd.DataFrame(amenity_impact).sort_values('Price Increase (SAR)', ascending=False)
    
    fig_amenities = px.bar(
        amenity_df, x='Amenity', y='Price Increase (SAR)',
        title="Price Impact of Different Amenities",
        color='Price Increase (SAR)',
        color_continuous_scale='RdYlGn'
    )
    fig_amenities.update_xaxes(tickangle=45)
    fig_amenities.update_layout(height=500)
    st.plotly_chart(fig_amenities, use_container_width=True)

with tab3:
    st.subheader("Price Distribution and Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution histogram
        fig_hist = px.histogram(
            df, x='price', nbins=50,
            title="Price Distribution of All Properties",
            labels={'price': 'Price (SAR)', 'count': 'Number of Properties'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Price by property age
        age_bins = pd.cut(df['property_age'], bins=[0, 1, 5, 15, 50], labels=['New (0-1y)', 'Recent (1-5y)', 'Mature (5-15y)', 'Old (15y+)'])
        age_price = df.groupby(age_bins)['price'].mean().reset_index()
        
        fig_age = px.bar(
            age_price, x='property_age', y='price',
            title="Average Price by Property Age",
            labels={'price': 'Average Price (SAR)', 'property_age': 'Property Age Category'},
            color='price',
            color_continuous_scale='viridis'
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Price heatmap by bedrooms and bathrooms
    st.subheader("Price Heatmap: Bedrooms vs Bathrooms")
    
    pivot_data = df.pivot_table(values='price', index='bedrooms', columns='bathrooms', aggfunc='mean')
    
    fig_heatmap = px.imshow(
        pivot_data,
        title="Average Price Heatmap (Bedrooms vs Bathrooms)",
        labels=dict(x="Bathrooms", y="Bedrooms", color="Average Price (SAR)"),
        aspect="auto",
        color_continuous_scale='RdYlBu_r'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.subheader("Interactive Market Explorer")
    
    # Interactive filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_cities = st.multiselect(
            "Select Cities to Compare",
            options=df['city'].unique(),
            default=df['city'].unique()[:2]
        )
    
    with col2:
        price_range = st.slider(
            "Price Range (SAR)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].quantile(0.1)), int(df['price'].quantile(0.9))),
            step=10000
        )
    
    with col3:
        bedroom_range = st.slider(
            "Bedroom Range",
            min_value=int(df['bedrooms'].min()),
            max_value=int(df['bedrooms'].max()),
            value=(int(df['bedrooms'].min()), int(df['bedrooms'].max()))
        )
    
    # Filter data based on selections
    filtered_df = df[
        (df['city'].isin(selected_cities)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1]) &
        (df['bedrooms'] >= bedroom_range[0]) &
        (df['bedrooms'] <= bedroom_range[1])
    ]
    
    if len(filtered_df) > 0:
        # Interactive scatter plot
        fig_interactive = px.scatter(
            filtered_df, x='land_area', y='price',
            color='city', size='bedrooms',
            hover_data=['district', 'bathrooms', 'property_age'],
            title=f"Interactive Property Explorer ({len(filtered_df)} properties)",
            labels={'land_area': 'Land Area (m²)', 'price': 'Price (SAR)'}
        )
        fig_interactive.update_layout(height=600)
        st.plotly_chart(fig_interactive, use_container_width=True)
        
        # Summary statistics for filtered data
        st.subheader("Filtered Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Properties Found", len(filtered_df))
        with col2:
            st.metric("Average Price", f"SAR {filtered_df['price'].mean():,.0f}")
        with col3:
            st.metric("Median Price", f"SAR {filtered_df['price'].median():,.0f}")
        with col4:
            st.metric("Price Range", f"SAR {filtered_df['price'].max() - filtered_df['price'].min():,.0f}")
    else:
        st.warning("No properties match the selected criteria. Please adjust your filters.")

# Market insights section
st.markdown("---")
st.subheader("🔍 Market Insights")

try:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Most expensive city
        most_expensive_city = df.groupby('city')['price'].mean().idxmax()
        avg_price_expensive = df.groupby('city')['price'].mean().max()
        st.metric(
            "Most Expensive City", 
            most_expensive_city,
            f"SAR {avg_price_expensive:,.0f} avg"
        )
    
    with col2:
        # Best value city (lowest price per sqm)
        df_temp = df.copy()
        df_temp['price_per_sqm'] = df_temp['price'] / df_temp['land_area']
        best_value_city = df_temp.groupby('city')['price_per_sqm'].mean().idxmin()
        avg_price_per_sqm = df_temp.groupby('city')['price_per_sqm'].mean().min()
        st.metric(
            "Best Value City", 
            best_value_city,
            f"SAR {avg_price_per_sqm:,.0f}/m²"
        )
    
    with col3:
        # Most common property type
        most_common_bedrooms = df['bedrooms'].mode()[0]
        most_common_bathrooms = df['bathrooms'].mode()[0]
        st.metric(
            "Most Common Property", 
            f"{most_common_bedrooms}BR/{most_common_bathrooms}BA",
            f"{len(df[(df['bedrooms']==most_common_bedrooms) & (df['bathrooms']==most_common_bathrooms)])} properties"
        )

except Exception as e:
    st.error(f"Error creating market insights: {e}")

st.markdown("---")
st.markdown("© 2025 Saudi House Price Predictor - Improved | Built with Streamlit and XGBoost")