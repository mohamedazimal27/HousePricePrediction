# 🏠 Saudi Housing Price Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)](https://xgboost.readthedocs.io/)

A machine learning system for predicting housing prices in Saudi Arabia using real estate data. This project achieves **81.07% accuracy** (R² score) in predicting property values based on key features.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Web Application](#-web-application)
- [Project Structure](#-project-structure)

## 🎯 Project Overview

This project predicts housing prices in Saudi Arabia using machine learning techniques. It processes real Saudi real estate data, applies advanced feature engineering, and trains an XGBoost model to predict property values with high accuracy.

### Key Features:
- **81.07% accuracy** (R² score) on test data
- **Real Saudi housing data** with English translation
- **Streamlit web interface** for easy interaction
- **Advanced feature engineering** with domain expertise
- **Robust data processing** with outlier handling

## 📊 Dataset

### Data Source
The dataset contains 3,718 Saudi property listings with 23 features including:
- **Location data**: City, district, front direction
- **Property features**: Bedrooms, bathrooms, land area, property age
- **Amenities**: Garage, driver room, maid room, air conditioning, etc.
- **Price**: Target variable in Saudi Riyal (SAR)

### Data Preprocessing
- **English translation** of all categorical values
- **Missing value handling** using median imputation
- **Outlier removal** for price values (5th to 95th percentile)
- **Feature engineering** to create new predictive features

## 📈 Model Performance

### Current Model (81.07% accuracy) ⭐
The model achieves high accuracy through:
- **Advanced feature engineering**:
  - Total rooms and room density calculations
  - Luxury score based on amenities
  - Price per square meter
  - Property age and size categories
- **Robust data processing** with outlier handling
- **Enhanced categorical encoding** for location features
- **RobustScaler** for better outlier resistance
- **Cross-validation** for reliable performance assessment

### Performance Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | **81.07%** | Model accuracy |
| **RMSE** | 18,767 SAR | Root mean square error |
| **MAE** | 8,026 SAR | Mean absolute error |
| **MAPE** | 10.57% | Mean absolute percentage error |

### Feature Importance (Top 10)
1. **land_area** - ~40%
2. **bedrooms** - ~20%
3. **bathrooms** - ~15%
4. **property_age** - ~10%
5. **living_rooms** - ~5%
6. **garage** - ~3%
7. **furnished** - ~2%
8. **air_conditioning** - ~1%
9. **driver_room** - ~1%
10. **maid_room** - ~1%

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
```bash
git clone https://github.com/mohamedazimal27/HousePricePrediction.git
cd HousePricePrediction
```

2. **Create Virtual Environment** (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training the Model
```bash
# Train the model
python train_model.py
```

### Checking Model Accuracy
```bash
python check_model.py
```

### Running Tests
```bash
python test_saudi_pipeline.py
```

## 🌐 Web Application

### Launch the App
```bash
# Launch the web application
streamlit run app.py
```

### Web Features
- **Interactive prediction form** with all property features
- **Real-time price predictions** in Saudi Riyal (SAR)
- **Data visualizations** showing price distributions and feature importance
- **Model information** and performance metrics
- **Responsive design** for desktop and mobile

Access the application at: `http://localhost:8501`

## 📁 Project Structure

```
ml/
├── README.md
├── requirements.txt
├── train_model.py                     # Model training script
├── check_model.py                     # Model accuracy verification
├── test_saudi_pipeline.py             # Test suite
├── app.py                             # Web application interface
├── style.css                          # Web styling
├── data/
│   └── processed/
│       └── saudi_housing_english.csv  # Main dataset
├── models/
│   └── saved/
│       ├── model.pkl                  # Trained XGBoost model
│       ├── scaler.pkl                 # RobustScaler for features
│       ├── features.pkl               # Feature list
│       └── encoders.pkl               # Categorical encoders
└── outputs/
    └── reports/
```

## 📧 Contact

Mohamed Azim Al - mohamedazimal27@gmail.com

Project Link: [https://github.com/mohamedazimal27/HousePricePrediction](https://github.com/mohamedazimal27/HousePricePrediction)

## 📃 License

This project is licensed under the MIT License.
