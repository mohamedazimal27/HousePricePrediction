# 🏠 Saudi Housing Price Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange)](https://xgboost.readthedocs.io/)

A machine learning system for predicting housing prices in Saudi Arabia using real estate data. This project achieves **77.47% accuracy** (R² score) in predicting property values based on key features.

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
- **77.47% accuracy** (R² score) on test data
- **Real Saudi housing data** with English translation
- **Streamlit web interface** for easy interaction
- **Comprehensive feature engineering** for improved predictions

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

### Presentation Model (77.47% accuracy) ⭐
The enhanced model significantly improved accuracy through:
- **Advanced feature engineering**:
  - Total rooms calculation
  - Luxury score based on amenities
  - Price per square meter
- **Categorical encoding** for city, district, and direction
- **Feature scaling** with StandardScaler
- **Model optimization** with XGBoost

### Performance Metrics
| Model | R² Score | RMSE (SAR) | MAE (SAR) | Error Rate |
|-------|----------|------------|-----------|-------------|
| Presentation | 77.47% | 15,716 | 7,566 | 19.60% |
| Original | 28.36% | 52,147 | 19,735 | 59.67% |

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
# Train the presentation model (recommended)
python train_model_for_presentation.py
```

### Checking Model Accuracy
```bash
python check_presentation_model.py
```

### Running Tests
```bash
python test_saudi_pipeline.py
```

## 🌐 Web Application

### Launch the App
```bash
# English interface (recommended)
streamlit run app_english.py
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
├── train_model_for_presentation.py    # Training script for presentation
├── check_presentation_model.py        # Model accuracy verification
├── test_saudi_pipeline.py             # Test suite
├── app_english.py                     # English web interface
├── style.css                          # Web styling
├── data/
│   └── processed/
│       └── saudi_housing_english.csv  # Main dataset
├── models/
│   └── saved/
│       ├── presentation_model.pkl     # Trained model
│       ├── presentation_scaler.pkl    # Feature scaler
│       ├── presentation_features.pkl  # Feature list
│       └── presentation_encoders.pkl  # Categorical encoders
└── outputs/
    └── reports/
```

## 📧 Contact

Mohamed Azim Al - mohamedazimal27@gmail.com

Project Link: [https://github.com/mohamedazimal27/HousePricePrediction](https://github.com/mohamedazimal27/HousePricePrediction)

## 📃 License

This project is licensed under the MIT License.
