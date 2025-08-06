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
    return pd.read_csv("saudi_housing_english.csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_saudi_house_price_model.pkl")

model = load_model()

# Load feature info
@st.cache_resource
def load_features():
    return joblib.load("saudi_model_features.pkl")

feature_info = load_features()
model_features = feature_info['features']

# --- Sidebar English Inputs ---
bedrooms = st.sidebar.slider("Bedrooms 🛏", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.slider("Bathrooms 🚿", min_value=1, max_value=10, value=2)
duplex = st.sidebar.selectbox("Property Type 🏠", options=["Duplex", "Non-Duplex"])
district = st.sidebar.selectbox("District 📍", options=sorted(df["district"].unique()))
property_age = st.sidebar.slider("Property Age 🏗️ (years)", min_value=0, max_value=50, value=5)
area = st.sidebar.slider("Land Area 📐 (m²)", min_value=100, max_value=1000, value=300)
living_rooms = st.sidebar.slider("Living Rooms 🛋", min_value=0, max_value=5, value=1)
garage = st.sidebar.selectbox("Garage 🚗", options=[0, 1, 2, 3])
driver_room = st.sidebar.selectbox("Driver Room 👨‍💼", options=[0, 1])
maid_room = st.sidebar.selectbox("Maid Room 👩‍💼", options=[0, 1])
furnished = st.sidebar.selectbox("Furnished 🪑", options=[0, 1])
ac = st.sidebar.selectbox("Air Conditioning ❄️", options=[0, 1])

# --- Encoding values ---
duplex_encoded = 1 if duplex == "Duplex" else 0

# Create district mapping for encoding
district_mapping = {name: idx for idx, name in enumerate(df["district"].astype("category").cat.categories)}
district_encoded = district_mapping.get(district, 0)

# Prepare input array based on model features
input_dict = {
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'land_area': area,
    'property_age': property_age,
    'living_rooms': living_rooms,
    'garage': garage,
    'driver_room': driver_room,
    'maid_room': maid_room,
    'furnished': furnished,
    'air_conditioning': ac,
    'duplex': duplex_encoded
}

# Create input array in correct order
input_data = np.array([[input_dict[feature] for feature in model_features]])

# Predict
prediction = model.predict(input_data)[0]
formatted_prediction = "SAR {:,.2f}".format(prediction)

# Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 Your Input Data")
    input_df = pd.DataFrame({
        'Feature': ['Bedrooms', 'Bathrooms', 'Property Type', 'District', 
                   'Property Age', 'Land Area (m²)', 'Living Rooms', 'Garage',
                   'Driver Room', 'Maid Room', 'Furnished', 'Air Conditioning'],
        'Value': [bedrooms, bathrooms, duplex, district, property_age, area,
                 living_rooms, garage, driver_room, maid_room, 
                 'Yes' if furnished else 'No', 'Yes' if ac else 'No']
    })
    st.table(input_df)

with col2:
    st.subheader("🔮 Predicted Price")
    st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
    st.info("This predicted price is based on your input and the trained model.")

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
    feature_importance = pd.DataFrame({
        'feature': model_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax2)
    ax2.set_title("Feature Importance")
    ax2.set_xlabel("Importance")
    ax2.set_ylabel("Features")
    st.pyplot(fig2)

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
