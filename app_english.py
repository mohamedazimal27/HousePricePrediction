#!/usr/bin/env python3
"""
Streamlit app for Saudi housing price prediction (English version)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder

# Set Streamlit config
st.set_page_config(page_title="🏡 Saudi House Price Predictor", layout="wide", page_icon="🏠")

# Load CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Title & Header
st.title("🏡 Saudi House Price Predictor")
st.markdown("Using XGBoost and Streamlit - Predict house prices based on your inputs!")

# Sidebar Inputs
st.sidebar.header("🔍 Enter Property Details")

# Load English dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/saudi_housing_english.csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/saved/presentation_model.pkl")

model = load_model()

# Load feature info and encoders
@st.cache_resource
def load_features_and_encoders():
    features = joblib.load("models/saved/presentation_features.pkl")
    encoders = joblib.load("models/saved/presentation_encoders.pkl")
    scaler = joblib.load("models/saved/presentation_scaler.pkl")
    return features, encoders, scaler

features, encoders, scaler = load_features_and_encoders()

# --- Sidebar English Inputs ---
# Get actual data ranges for validation
max_bedrooms = int(df["bedrooms"].max())
max_bathrooms = int(df["bathrooms"].max())
max_property_age = int(df["property_age"].max())
max_land_area = int(df["land_area"].max())
max_living_rooms = int(df["living_rooms"].max())
max_garage = int(df["garage"].max())

bedrooms = st.sidebar.slider("Bedrooms 🛏", min_value=1, max_value=max_bedrooms, value=min(3, max_bedrooms))
bathrooms = st.sidebar.slider("Bathrooms 🚿", min_value=1, max_value=max_bathrooms, value=min(2, max_bathrooms))
duplex = st.sidebar.selectbox("Property Type 🏠", options=["Duplex", "Non-Duplex"])
district = st.sidebar.selectbox("District 📍", options=sorted(df["district"].unique()))
city = st.sidebar.selectbox("City 🏙️", options=sorted(df["city"].unique()))
front_direction = st.sidebar.selectbox("Front Direction 🧭", options=sorted(df["front_direction"].unique()))
property_age = st.sidebar.slider("Property Age 🏗️ (years)", min_value=0, max_value=max_property_age, value=min(5, max_property_age))
area = st.sidebar.slider("Land Area 📐 (m²)", min_value=100, max_value=min(max_land_area, 5000), value=min(300, max_land_area))
living_rooms = st.sidebar.slider("Living Rooms 🛋", min_value=1, max_value=max_living_rooms, value=min(1, max_living_rooms))
garage = st.sidebar.selectbox("Garage 🚗", options=list(range(max_garage + 1)))
driver_room = st.sidebar.selectbox("Driver Room 👨‍💼", options=[0, 1])
maid_room = st.sidebar.selectbox("Maid Room 👩‍💼", options=[0, 1])
furnished = st.sidebar.selectbox("Furnished 🪑", options=[0, 1])
ac = st.sidebar.selectbox("Air Conditioning ❄️", options=[0, 1])
roof = st.sidebar.selectbox("Roof 🏠", options=[0, 1])
pool = st.sidebar.selectbox("Pool 🏊", options=[0, 1])
front_yard = st.sidebar.selectbox("Front Yard 🌳", options=[0, 1])
basement = st.sidebar.selectbox("Basement 🏠", options=[0, 1])
stairs = st.sidebar.selectbox("Stairs 🪜", options=[0, 1])
elevator = st.sidebar.selectbox("Elevator 🚠", options=[0, 1])
fireplace = st.sidebar.selectbox("Fireplace 🔥", options=[0, 1])

# --- Encoding values ---
duplex_encoded = 1 if duplex == "Duplex" else 0

# Encode categorical variables using the saved encoders
try:
    city_encoded = encoders['city'].transform([city])[0]
except ValueError:
    city_encoded = 0  # Default for unknown categories

try:
    district_encoded = encoders['district'].transform([district])[0]
except ValueError:
    district_encoded = 0  # Default for unknown categories

try:
    front_direction_encoded = encoders['front_direction'].transform([front_direction])[0]
except ValueError:
    front_direction_encoded = 0  # Default for unknown categories

# Calculate derived features
total_rooms = bedrooms + bathrooms + living_rooms
luxury_score = garage + driver_room + maid_room + furnished + ac + roof + pool + front_yard + basement + duplex_encoded + stairs + elevator + fireplace

# Prepare input array based on model features
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

# Create input array in correct order and scale it
input_data = np.array([[input_dict[feature] for feature in features]])
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)[0]
formatted_prediction = "SAR {:,.2f}".format(prediction)

# Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 Your Input Data")
    
    # Check for extreme values
    extreme_warnings = []
    if bedrooms >= max_bedrooms * 0.9:
        extreme_warnings.append(f"Bedrooms ({bedrooms}) is near maximum ({max_bedrooms})")
    if bathrooms >= max_bathrooms * 0.9:
        extreme_warnings.append(f"Bathrooms ({bathrooms}) is near maximum ({max_bathrooms})")
    if area >= min(max_land_area, 5000) * 0.9:
        extreme_warnings.append(f"Land area ({area} m²) is near maximum")
    
    if extreme_warnings:
        st.warning("⚠️ **Extreme Values Detected:**\n" + "\n".join([f"- {warning}" for warning in extreme_warnings]))
    
    # Create input dataframe with proper data types
    input_df = pd.DataFrame({
        'Feature': ['City', 'District', 'Front Direction', 'Bedrooms', 'Bathrooms', 'Property Type', 
                   'Property Age', 'Land Area (m²)', 'Living Rooms', 'Garage',
                   'Driver Room', 'Maid Room', 'Furnished', 'Air Conditioning',
                   'Roof', 'Pool', 'Front Yard', 'Basement', 'Stairs', 'Elevator', 'Fireplace'],
        'Value': [str(city), str(district), str(front_direction), str(bedrooms), str(bathrooms), str(duplex), str(property_age), str(area),
                 str(living_rooms), str(garage), str(driver_room), str(maid_room), 
                 'Yes' if furnished else 'No', 'Yes' if ac else 'No',
                 'Yes' if roof else 'No', 'Yes' if pool else 'No', 'Yes' if front_yard else 'No',
                 'Yes' if basement else 'No', 'Yes' if stairs else 'No', 'Yes' if elevator else 'No',
                 'Yes' if fireplace else 'No']
    })
    st.table(input_df)

with col2:
    st.subheader("🔮 Predicted Price")
    st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
    st.info("This predicted price is based on your input and the trained model.")
    
    # Display data ranges
    with st.expander("📊 Data Ranges Used for Training"):
        st.write(f"- **Bedrooms**: 1-{max_bedrooms}")
        st.write(f"- **Bathrooms**: 1-{max_bathrooms}")
        st.write(f"- **Land Area**: 100-{min(max_land_area, 5000)} m²")
        st.write(f"- **Property Age**: 0-{max_property_age} years")
        st.write(f"- **Living Rooms**: 1-{max_living_rooms}")
        st.write(f"- **Garage**: 0-{max_garage}")

# Plots
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### House Price Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["price"], kde=True, ax=ax1)
    ax1.set_xlabel("Price (SAR)")
    ax1.set_ylabel("Number of Properties")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📈 Feature Importance")
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature', ax=ax2)
        ax2.set_title("Feature Importance")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Features")
        st.pyplot(fig2)
    else:
        st.info("Feature importance not available for this model type")

# Dataset sample
st.markdown("---")
st.subheader("🧾 Data Preview")
st.write(df.head())

# City distribution
st.markdown("---")
st.subheader("📍 Properties by City")
city_counts = df['city'].value_counts()
fig3, ax3 = plt.subplots(figsize=(8, 4))
city_counts.plot(kind='bar', ax=ax3)
ax3.set_title("Number of Properties by City")
ax3.set_xlabel("City")
ax3.set_ylabel("Count")
plt.xticks(rotation=45)
st.pyplot(fig3)

st.markdown("---")
st.markdown("© 2025 Saudi House Price Predictor | Built with Streamlit and XGBoost")
