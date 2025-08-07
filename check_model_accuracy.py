#!/usr/bin/env python3
"""
Comprehensive Model Accuracy Checker for Saudi House Price Prediction
This script evaluates the accuracy of all existing models without modifying any code.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed dataset."""
    try:
        df = pd.read_csv('data/processed/saudi_housing_english.csv')
        print(f"✅ Loaded dataset: {len(df):,} samples")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def check_original_model(df):
    """Check accuracy of the original XGBoost model."""
    print("\n" + "="*50)
    print("1. ORIGINAL MODEL ACCURACY")
    print("="*50)
    
    try:
        # Load model components
        model = joblib.load('models/saved/xgboost_saudi_house_price_model.pkl')
        feature_info = joblib.load('models/saved/saudi_model_features.pkl')
        
        features = feature_info['features']
        X = df[features].copy()
        y = df['price']
        
        # Clean data
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            print("❌ No valid data after cleaning")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"✅ Model loaded successfully")
        print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"   RMSE: {rmse:,.0f} SAR")
        print(f"   MAE: {mae:,.0f} SAR")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Mean Price: {y_clean.mean():,.0f} SAR")
        print(f"   Error Rate: {(rmse/y_clean.mean()*100):.2f}%")
        print(f"   Samples used: {len(X_clean):,}")
        
        return {
            'model': 'Original',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'samples': len(X_clean)
        }
        
    except Exception as e:
        print(f"❌ Original model error: {e}")
        return None

def check_enhanced_model(df):
    """Check accuracy of the enhanced model."""
    print("\n" + "="*50)
    print("2. ENHANCED MODEL ACCURACY")
    print("="*50)
    
    try:
        # Load model components
        model = joblib.load('models/saved/enhanced_saudi_house_price_model.pkl')
        scaler = joblib.load('models/saved/enhanced_model_scaler.pkl')
        features = joblib.load('models/saved/enhanced_model_features.pkl')
        encoders = joblib.load('models/saved/enhanced_model_encoders.pkl')
        
        # Prepare data with same preprocessing as training
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['city', 'district', 'front_direction', 'age_category']
        for col in categorical_cols:
            if col in df_processed.columns and col in encoders:
                known_categories = encoders[col].classes_
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_categories else 'Other'
                )
                df_processed[col] = encoders[col].transform(df_processed[col])
        
        # Create luxury score
        luxury_features = ['has_garage', 'has_elevator', 'has_pool', 'has_garden', 'has_ac', 'has_balcony']
        existing_features = [f for f in luxury_features if f in df_processed.columns]
        if existing_features:
            df_processed['luxury_score'] = df_processed[existing_features].sum(axis=1)
        else:
            df_processed['luxury_score'] = 0
        
        # Select features
        available_features = [f for f in features if f in df_processed.columns]
        X = df_processed[available_features].copy()
        y = df_processed['price']
        
        # Handle missing values and outliers
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        Q1 = y.quantile(0.05)
        Q3 = y.quantile(0.95)
        mask = (y >= Q1) & (y <= Q3)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("❌ No valid data after preprocessing")
            return None
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"✅ Enhanced model loaded successfully")
        print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"   RMSE: {rmse:,.0f} SAR")
        print(f"   MAE: {mae:,.0f} SAR")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Mean Price: {y.mean():,.0f} SAR")
        print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
        print(f"   Samples used: {len(X):,}")
        
        return {
            'model': 'Enhanced',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'samples': len(X)
        }
        
    except Exception as e:
        print(f"❌ Enhanced model error: {e}")
        return None

def check_improved_model(df):
    """Check accuracy of the improved model."""
    print("\n" + "="*50)
    print("3. IMPROVED MODEL ACCURACY")
    print("="*50)
    
    try:
        # Load model components
        model = joblib.load('models/saved/improved_saudi_house_price_model.pkl')
        scaler = joblib.load('models/saved/improved_model_scaler.pkl')
        features = joblib.load('models/saved/improved_model_features.pkl')
        encoders = joblib.load('models/saved/improved_model_encoders.pkl')
        
        # Prepare data
        df_processed = df.copy()
        
        # Feature engineering
        df_processed['price_per_sqm'] = df_processed['price'] / df_processed['land_area']
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['living_rooms']
        df_processed['bathroom_ratio'] = df_processed['bathrooms'] / (df_processed['bedrooms'] + 1)
        
        luxury_features = ['has_garage', 'has_elevator', 'has_pool', 'has_garden', 'has_ac', 'has_balcony']
        existing_features = [f for f in luxury_features if f in df_processed.columns]
        df_processed['luxury_score'] = df_processed[existing_features].sum(axis=1) if existing_features else 0
        
        # Handle categorical variables
        categorical_cols = ['city', 'district', 'front_direction', 'age_category']
        for col in categorical_cols:
            if col in df_processed.columns and col in encoders:
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = encoders[col].transform(df_processed[col])
        
        # Select features
        X = df_processed[features].copy()
        y = df_processed['price']
        
        # Clean data
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        Q1 = y.quantile(0.01)
        Q3 = y.quantile(0.99)
        mask = (y >= Q1) & (y <= Q3)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("❌ No valid data after preprocessing")
            return None
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"✅ Improved model loaded successfully")
        print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"   RMSE: {rmse:,.0f} SAR")
        print(f"   MAE: {mae:,.0f} SAR")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Mean Price: {y.mean():,.0f} SAR")
        print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
        print(f"   Samples used: {len(X):,}")
        
        return {
            'model': 'Improved',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'samples': len(X)
        }
        
    except Exception as e:
        print(f"❌ Improved model error: {e}")
        return None

def check_ultra_accuracy_model(df):
    """Check accuracy of the ultra-high accuracy model."""
    print("\n" + "="*50)
    print("4. ULTRA-HIGH ACCURACY MODEL")
    print("="*50)
    
    try:
        # Load model components
        model = joblib.load('models/saved/ultra_accuracy_saudi_house_price_model.pkl')
        scaler = joblib.load('models/saved/ultra_accuracy_model_scaler.pkl')
        features = joblib.load('models/saved/ultra_accuracy_model_features.pkl')
        encoders = joblib.load('models/saved/ultra_accuracy_model_encoders.pkl')
        selector = joblib.load('models/saved/ultra_accuracy_feature_selector.pkl')
        
        # Prepare data with advanced preprocessing
        df_processed = df.copy()
        
        # Advanced feature engineering
        df_processed['price_per_sqm'] = df_processed['price'] / df_processed['land_area']
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['living_rooms']
        df_processed['bathroom_ratio'] = df_processed['bathrooms'] / (df_processed['bedrooms'] + 1)
        df_processed['room_density'] = df_processed['total_rooms'] / df_processed['land_area']
        df_processed['luxury_score'] = (
            df_processed.get('has_garage', 0) + 
            df_processed.get('has_elevator', 0) + 
            df_processed.get('has_pool', 0) + 
            df_processed.get('has_garden', 0) + 
            df_processed.get('has_ac', 0) + 
            df_processed.get('has_balcony', 0)
        )
        
        # Interaction features
        df_processed['bedrooms_bathrooms'] = df_processed['bedrooms'] * df_processed['bathrooms']
        df_processed['land_area_rooms'] = df_processed['land_area'] * df_processed['total_rooms']
        df_processed['price_land_ratio'] = df_processed['price'] / (df_processed['land_area'] + 1)
        
        # Handle categorical variables
        categorical_cols = ['city', 'district', 'front_direction', 'age_category']
        for col in categorical_cols:
            if col in df_processed.columns and col in encoders:
                # Frequency encoding
                freq_map = df_processed[col].value_counts().to_dict()
                df_processed[f'{col}_freq'] = df_processed[col].map(freq_map)
                
                # Label encoding
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = encoders[col].transform(df_processed[col])
        
        # Select features
        X = df_processed[features].copy()
        y = df_processed['price']
        
        # Clean data
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        Q1 = y.quantile(0.05)
        Q3 = y.quantile(0.95)
        mask = (y >= Q1) & (y <= Q3)
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print("❌ No valid data after preprocessing")
            return None
        
        # Split, scale, and select features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Make predictions
        y_pred = model.predict(X_test_selected)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"✅ Ultra-high accuracy model loaded successfully")
        print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"   RMSE: {rmse:,.0f} SAR")
        print(f"   MAE: {mae:,.0f} SAR")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Mean Price: {y.mean():,.0f} SAR")
        print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
        print(f"   Samples used: {len(X):,}")
        print(f"   Features used: {len(features)}")
        
        return {
            'model': 'Ultra-High Accuracy',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'samples': len(X),
            'features': len(features)
        }
        
    except Exception as e:
        print(f"❌ Ultra-high accuracy model error: {e}")
        return None

def print_summary(results):
    """Print a summary comparison of all models."""
    print("\n" + "="*70)
    print("MODEL ACCURACY SUMMARY")
    print("="*70)
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("❌ No models could be evaluated")
        return
    
    # Sort by R² score (descending)
    valid_results.sort(key=lambda x: x['r2'], reverse=True)
    
    print(f"{'Model':<20} {'R² Score':<12} {'RMSE (SAR)':<12} {'MAE (SAR)':<12} {'MAPE (%)':<10}")
    print("-" * 70)
    
    for result in valid_results:
        print(f"{result['model']:<20} {result['r2']:<12.4f} {result['rmse']:<12,.0f} "
              f"{result['mae']:<12,.0f} {result['mape']:<10.2f}")
    
    # Best model
    best = valid_results[0]
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best['model']}")
    print(f"   Accuracy: {best['r2']*100:.2f}%")
    print(f"   Average Error: {best['mae']:,.0f} SAR")
    print(f"   Error Rate: {best['mape']:.2f}%")
    print("="*70)

def main():
    """Main function to check all model accuracies."""
    print("🔍 SAUDI HOUSE PRICE MODEL ACCURACY CHECKER")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Print dataset summary
    print("\n📊 DATASET SUMMARY:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Price range: SAR {df['price'].min():,} - SAR {df['price'].max():,}")
    print(f"   Mean price: SAR {df['price'].mean():,.0f}")
    print(f"   Median price: SAR {df['price'].median():,.0f}")
    print(f"   Standard deviation: SAR {df['price'].std():,.0f}")
    
    # Check all models
    results = []
    results.append(check_original_model(df))
    results.append(check_enhanced_model(df))
    results.append(check_improved_model(df))
    results.append(check_ultra_accuracy_model(df))
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Accuracy check completed!")

if __name__ == "__main__":
    main()
