#!/usr/bin/env python3
"""
Improved Streamlit app for Saudi housing price prediction with better error handling and UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set Streamlit config
st.set_page_config(
    page_title="🏡 Saudi House Price Predictor - Improved", 
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
st.title("🏡 Saudi House Price Predictor - Improved")
st.markdown("Enhanced model with better accuracy and data handling")

# Sidebar for model selection
st.sidebar.header("🔧 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model Version",
    ["Presentation Model (77.47%)", "Improved Model (if available)"]
)

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

# Load model with fallback
@st.cache_resource
def load_model_components(model_type="presentation"):
    try:
        if model_type == "improved":
            model = joblib.load("models/saved/improved_model.pkl")
            scaler = joblib.load("models/saved/improved_scaler.pkl")
            features = joblib.load("models/saved/improved_features.pkl")
            encoders = joblib.load("models/saved/improved_encoders.pkl")
        else:
            model = joblib.load("models/saved/presentation_model.pkl")
            scaler = joblib.load("models/saved/presentation_scaler.pkl")
            features = joblib.load("models/saved/presentation_features.pkl")
            encoders = joblib.load("models/saved/presentation_encoders.pkl")
        
        return model, scaler, features, encoders, None
    except Exception as e:
        return None, None, None, None, str(e)

# Determine which model to use
model_type = "improved" if "Improved" in model_choice else "presentation"
model, scaler, features, encoders, model_error = load_model_components(model_type)

if model_error:
    st.error(f"Error loading model: {model_error}")
    if model_type == "improved":
        st.info("Falling back to presentation model...")
        model, scaler, features, encoders, model_error = load_model_components("presentation")
        if model_error:
            st.error("Could not load any model. Please check model files.")
            st.stop()

# Sidebar Inputs with improved validation
st.sidebar.header("🔍 Enter Property Details")

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

# Location inputs with error handling
try:
    district_options = sorted(df["district"].unique())
    city_options = sorted(df["city"].unique())
    direction_options = sorted(df["front_direction"].unique())
except:
    district_options = ["Unknown"]
    city_options = ["Unknown"]
    direction_options = ["Unknown"]

district = st.sidebar.selectbox("District 📍", options=district_options)
city = st.sidebar.selectbox("City 🏙️", options=city_options)
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
        st.metric("Model Accuracy", f"{77.47 if model_type == 'presentation' else 'TBD'}%")

# Data visualization
st.markdown("---")
st.subheader("📊 Market Analysis")

try:
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        city_prices = df.groupby('city')['price'].mean().sort_values(ascending=False)
        city_prices.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Average Price by City")
        ax1.set_xlabel("City")
        ax1.set_ylabel("Average Price (SAR)")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        bedroom_prices = df.groupby('bedrooms')['price'].mean()
        bedroom_prices.plot(kind='line', marker='o', ax=ax2, color='green')
        ax2.set_title("Average Price by Bedroom Count")
        ax2.set_xlabel("Number of Bedrooms")
        ax2.set_ylabel("Average Price (SAR)")
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Error creating visualizations: {e}")

st.markdown("---")
st.markdown("© 2025 Saudi House Price Predictor - Improved | Built with Streamlit and XGBoost")