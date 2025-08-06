import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import plot_importance

# Set Streamlit config
st.set_page_config(page_title="🏡 توقع سعر العقار", layout="wide", page_icon="🏠")

# Load CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Title & Header
st.title("🏡 تطبيق توقع أسعار العقارات في السعودية")
st.markdown("باستخدام XGBoost و Streamlit — توقع سعر العقار بناءً على مدخلاتك!")

# Sidebar Inputs
st.sidebar.header("🔍 أدخل معلومات العقار")

# Load Arabic dataset
@st.cache_data
def load_data():
    return pd.read_csv("SA_Aqar.csv")

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_house_price_model.pkl")

model = load_model()

# --- Sidebar Arabic Inputs ---
bedrooms = st.sidebar.slider("عدد غرف النوم 🛏", min_value=1, max_value=10, value=3)
duplex = st.sidebar.selectbox("نوع العقار 🏠", options=["دبلكس", "غير دبلكس"])
district = st.sidebar.selectbox("الحي 📍", options=sorted(df["district"].unique()))
property_age = st.sidebar.slider("عمر العقار 🏗️ (بالسنوات)", min_value=0, max_value=50, value=5)
area = st.sidebar.slider("المساحة 📐 (م²)", min_value=100, max_value=1000, value=300)

# --- Encoding values ---
duplex_encoded = 1 if duplex == "دبلكس" else 0
district_mapping = {name: idx for idx, name in enumerate(df["district"].astype("category").cat.categories)}
district_encoded = district_mapping.get(district, 0)

# Prepare input
input_data = np.array([[bedrooms, duplex_encoded, district_encoded, property_age, area]])

# Predict
prediction = model.predict(input_data)[0]
formatted_prediction = "﷼ {:,.2f}".format(prediction)

# Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 بياناتك المدخلة")
    input_df = pd.DataFrame({
        'الميزة': ['عدد الغرف', 'نوع العقار', 'الحي', 'عمر العقار', 'المساحة (م²)'],
        'القيمة': [bedrooms, duplex, district, property_age, area]
    })
    st.table(input_df)

with col2:
    st.subheader("🔮 السعر المتوقع")
    st.markdown(f"<h2 style='color:#4CAF50;'>{formatted_prediction}</h2>", unsafe_allow_html=True)
    st.info("هذا السعر المتوقع بناءً على البيانات المدخلة والنموذج المدرب.")

# Plots
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### توزيع أسعار العقارات")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(df["price"], kde=True, ax=ax1)
    ax1.set_xlabel("السعر")
    ax1.set_ylabel("عدد العقارات")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📈 أهمية الخصائص في النموذج")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_importance(model, ax=ax2, max_num_features=5, color="#4285F4")
    st.pyplot(fig2)

# Dataset sample
st.markdown("---")
st.subheader("🧾 معاينة البيانات")
st.write(df.head())

st.markdown("---")
st.markdown("© 2025 توقع أسعار العقارات | بناء بواسطة Streamlit و XGBoost")
