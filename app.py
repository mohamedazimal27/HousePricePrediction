#!/usr/bin/env python3
"""
Streamlit app for Saudi housing price prediction
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

# Load dataset
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

# --- Sidebar Inputs ---
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
property_age = st.sidebar.slider("Property Age (Years) 📅", min_value=0, max_value=max_property_age, value=5)
land_area = st.sidebar.slider("Land Area (sqm) 📐", min_value=50, max_value=max_land_area, value=200)
living_rooms = st.sidebar.slider("Living Rooms 🛋", min_value=0, max_value=max_living_rooms, value=1)
garage = st.sidebar.slider("Garage Spaces 🚗", min_value=0, max_value=max_garage, value=1)

# Binary features
driver_room = st.sidebar.checkbox("Driver Room 👨‍💼")
maid_room = st.sidebar.checkbox("Maid Room 👩‍💼")
furnished = st.sidebar.checkbox("Furnished 🪑")
air_conditioning = st.sidebar.checkbox("Air Conditioning ❄️")

# --- Prediction ---
if st.sidebar.button("💰 Predict Price", type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'land_area': [land_area],
        'property_age': [property_age],
        'living_rooms': [living_rooms],
        'garage': [garage],
        'driver_room': [1 if driver_room else 0],
        'maid_room': [1 if maid_room else 0],
        'furnished': [1 if furnished else 0],
        'air_conditioning': [1 if air_conditioning else 0],
        'duplex': [1 if duplex == "Duplex" else 0]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.success(f"🏠 **Predicted Price: SAR {prediction:,.0f}**")
    
    # Confidence interval (approximate)
    st.info(f"📊 Estimated Range: SAR {prediction * 0.8:,.0f} - SAR {prediction * 1.2:,.0f}")

# --- Main Content ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🎯 Model Info", "📋 About"])

with tab1:
    st.header("📊 Saudi Housing Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Avg Price", f"SAR {df['price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"SAR {df['price'].min():,.0f} - SAR {df['price'].max():,.0f}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

with tab2:
    st.header("📈 Data Visualizations")
    
    # Price distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price distribution
    axes[0, 0].hist(df['price'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Price Distribution (SAR)')
    axes[0, 0].set_xlabel('Price (SAR)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Bedrooms vs Price
    axes[0, 1].scatter(df['bedrooms'], df['price'], alpha=0.5, color='green')
    axes[0, 1].set_title('Bedrooms vs Price')
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price (SAR)')
    
    # Land Area vs Price
    axes[1, 0].scatter(df['land_area'], df['price'], alpha=0.5, color='orange')
    axes[1, 0].set_title('Land Area vs Price')
    axes[1, 0].set_xlabel('Land Area (sqm)')
    axes[1, 0].set_ylabel('Price (SAR)')
    
    # Property Age vs Price
    axes[1, 1].scatter(df['property_age'], df['price'], alpha=0.5, color='red')
    axes[1, 1].set_title('Property Age vs Price')
    axes[1, 1].set_xlabel('Property Age (Years)')
    axes[1, 1].set_ylabel('Price (SAR)')
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.header("🎯 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.write("**Algorithm:** XGBoost Regressor")
        st.write("**R² Score:** 0.28")
        st.write("**RMSE:** SAR 52,147")
        st.write("**Training Samples:** 2,974")
        st.write("**Test Samples:** 744")
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': model_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.header("📋 About")
    st.markdown("""
    ### 🏡 Saudi House Price Predictor
    
    This application uses machine learning to predict house prices in Saudi Arabia based on property features.
    
    **Features Used:**
    - Bedrooms, Bathrooms, Living Rooms
    - Land Area (sqm)
    - Property Age (years)
    - Garage Spaces
    - Amenities: Driver Room, Maid Room, Furnished, Air Conditioning
    - Property Type: Duplex/Non-Duplex
    
    **Model Details:**
    - **Algorithm:** XGBoost Regressor
    - **Data:** 3,718 Saudi properties
    - **Currency:** Saudi Riyal (SAR)
    - **Performance:** R² = 0.28
    
    **How to Use:**
    1. Use the sidebar to input property details
    2. Click "Predict Price" to get the estimated price
    3. Explore the tabs for data insights
    
    **Note:** Predictions are estimates based on historical data. Actual prices may vary.
    """)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for Saudi Real Estate Market*")
