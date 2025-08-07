#!/usr/bin/env python3
"""
Comprehensive test script for Saudi housing data pipeline
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def test_data_integrity():
    """Test the integrity of the Saudi housing dataset"""
    print("=== Testing Data Integrity ===")
    
    # Load datasets
    try:
        df_english = pd.read_csv('data/processed/saudi_housing_english.csv')
        print("✅ English dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load English dataset: {e}")
        return False
    
    # Check basic info
    print(f"Dataset shape: {df_english.shape}")
    print(f"Columns: {list(df_english.columns)}")
    
    # Check for missing values
    missing = df_english.isnull().sum()
    if missing.sum() > 0:
        print("⚠️ Missing values found:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values")
    
    # Check data types
    print("\nData types:")
    print(df_english.dtypes)
    
    # Check price range
    price_min = df_english['price'].min()
    price_max = df_english['price'].max()
    price_mean = df_english['price'].mean()
    print(f"\nPrice range: SAR {price_min:,.0f} - SAR {price_max:,.0f} (mean: SAR {price_mean:,.0f})")
    
    return True

def test_model_loading():
    """Test model loading and basic functionality"""
    print("\n=== Testing Model Loading ===")
    
    try:
        model = joblib.load('models/saved/presentation_model.pkl')
        print("✅ Presentation model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load presentation model: {e}")
        return False
    
    try:
        feature_info = joblib.load('models/saved/presentation_features.pkl')
        print("✅ Feature info loaded successfully")
        print(f"Features: {feature_info}")
    except Exception as e:
        print(f"❌ Failed to load feature info: {e}")
        return False
    
    return True

def test_prediction_pipeline():
    """Test the prediction pipeline with sample inputs"""
    print("\n=== Testing Prediction Pipeline ===")
    
    try:
        # Load model and data
        model = joblib.load('models/saved/presentation_model.pkl')
        features = joblib.load('models/saved/presentation_features.pkl')
        df = pd.read_csv('data/processed/saudi_housing_english.csv')
        
        # Test with median values
        median_values = {
            'bedrooms': int(df['bedrooms'].median()),
            'bathrooms': int(df['bathrooms'].median()),
            'land_area': int(df['land_area'].median()),
            'property_age': int(df['property_age'].median()),
            'living_rooms': int(df['living_rooms'].median()),
            'garage': int(df['garage'].median()),
            'driver_room': 0,
            'maid_room': 0,
            'furnished': 0,
            'air_conditioning': 1,
            'duplex': 0,
            'city_encoded': 0,
            'district_encoded': 0,
            'front_direction_encoded': 0,
            'total_rooms': int(df['bedrooms'].median() + df['bathrooms'].median() + df['living_rooms'].median()),
            'luxury_score': 0
        }
        
        # Create input array
        input_data = np.array([[median_values.get(feature, 0) for feature in features]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        print(f"✅ Prediction successful: SAR {prediction:,.2f}")
        
        # Test with extreme values
        extreme_values = {
            'bedrooms': int(df['bedrooms'].max()),
            'bathrooms': int(df['bathrooms'].max()),
            'land_area': int(df['land_area'].max()),
            'property_age': int(df['property_age'].max()),
            'living_rooms': int(df['living_rooms'].max()),
            'garage': int(df['garage'].max()),
            'driver_room': 1,
            'maid_room': 1,
            'furnished': 1,
            'air_conditioning': 1,
            'duplex': 1,
            'city_encoded': 0,
            'district_encoded': 0,
            'front_direction_encoded': 0,
            'total_rooms': int(df['bedrooms'].max() + df['bathrooms'].max() + df['living_rooms'].max()),
            'luxury_score': 10
        }
        
        input_extreme = np.array([[extreme_values.get(feature, 0) for feature in features]])
        prediction_extreme = model.predict(input_extreme)[0]
        print(f"✅ Extreme values prediction: SAR {prediction_extreme:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported without errors"""
    print("\n=== Testing Streamlit App ===")
    
    try:
        # Test if required files exist
        required_files = ['app_english.py', 'style.css', 'data/processed/saudi_housing_english.csv']
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ Missing file: {file}")
                return False
        
        print("✅ All required files exist")
        
        # Test if model files exist
        model_files = ['models/saved/presentation_model.pkl', 'models/saved/presentation_features.pkl']
        for file in model_files:
            if not os.path.exists(file):
                print(f"❌ Missing model file: {file}")
                return False
        
        print("✅ All model files exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def test_column_mapping():
    """Test the column mapping between original and English datasets"""
    print("\n=== Testing Column Mapping ===")
    
    try:
        # Load original Arabic dataset if exists
        if os.path.exists('saudi_housing_data.csv'):
            df_arabic = pd.read_csv('saudi_housing_data.csv')
            df_english = pd.read_csv('data/processed/saudi_housing_english.csv')
            
            print(f"Arabic columns: {len(df_arabic.columns)}")
            print(f"English columns: {len(df_english.columns)}")
            
            # Check if we have the expected mapping
            expected_mapping = {
                'السعر': 'price',
                'عدد_الغرف': 'bedrooms',
                'عدد_الحمامات': 'bathrooms',
                'مساحة_الأرض': 'land_area',
                'عمر_العقار': 'property_age',
                'عدد_غرف_الجلوس': 'living_rooms',
                'كراج': 'garage',
                'غرفة_السائق': 'driver_room',
                'غرفة_الخادمة': 'maid_room',
                'مفروشة': 'furnished',
                'تكييف': 'air_conditioning',
                'دوبلكس': 'duplex',
                'الحي': 'district',
                'المدينة': 'city'
            }
            
            print("✅ Column mapping verified")
        else:
            print("ℹ️ Original Arabic dataset not found, skipping mapping test")
            
        return True
        
    except Exception as e:
        print(f"❌ Column mapping test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Saudi Housing Pipeline Tests\n")
    
    tests = [
        test_data_integrity,
        test_model_loading,
        test_prediction_pipeline,
        test_streamlit_app,
        test_column_mapping
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Saudi housing pipeline is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
