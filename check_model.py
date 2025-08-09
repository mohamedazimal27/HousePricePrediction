#!/usr/bin/env python3
"""
Check the accuracy and performance of the Saudi housing prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== SAUDI HOUSING MODEL ACCURACY CHECK ===")
    
    # Load the model components
    try:
        model = joblib.load('models/saved/model.pkl')
        scaler = joblib.load('models/saved/scaler.pkl')
        features = joblib.load('models/saved/features.pkl')
        encoders = joblib.load('models/saved/encoders.pkl')
        print("✅ Model components loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        return
    
    # Load data
    try:
        df = pd.read_csv('data/processed/saudi_housing_english.csv')
        print("✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Apply same preprocessing as training
    print("\n=== APPLYING PREPROCESSING ===")
    
    # Data cleaning (same as training)
    initial_count = len(df)
    price_lower = df['price'].quantile(0.01)
    price_upper = df['price'].quantile(0.99)
    df = df[(df['price'] >= price_lower) & (df['price'] <= price_upper)]
    df = df[df['land_area'] >= 50]
    df = df[df['land_area'] <= 5000]
    df = df[df['bedrooms'] <= 8]
    
    # Feature engineering (same as training)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['living_rooms']
    luxury_features = ['garage', 'driver_room', 'maid_room', 'furnished', 
                      'air_conditioning', 'pool', 'front_yard', 'basement', 
                      'duplex', 'elevator', 'fireplace']
    df['luxury_score'] = df[luxury_features].sum(axis=1)
    df['price_per_sqm'] = df['price'] / df['land_area']
    df['room_density'] = df['total_rooms'] / df['land_area'] * 1000
    
    # Age and size categories
    df['age_category'] = pd.cut(df['property_age'], 
                               bins=[-1, 0, 5, 15, 50], 
                               labels=['new', 'recent', 'mature', 'old'])
    df['size_category'] = pd.cut(df['land_area'], 
                                bins=[0, 200, 400, 800, 5000], 
                                labels=['small', 'medium', 'large', 'xlarge'])
    
    # Categorical encoding
    categorical_cols = ['city', 'district', 'front_direction', 'age_category', 'size_category']
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            # Handle unknown categories
            known_categories = encoders[col].classes_
            df[col] = df[col].apply(
                lambda x: x if x in known_categories else 'Unknown'
            )
            df[f'{col}_encoded'] = encoders[col].transform(df[col])
    
    # Select features
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()
    y = df['price']
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    print(f"   Using {len(available_features)} features")
    print(f"   Samples after preprocessing: {len(X)}")
    
    # Split data (same random state as training)
    price_quartiles = pd.qcut(y, q=4, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_quartiles
    )
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("\n=== MODEL EVALUATION ===")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
    print(f"   RMSE: {rmse:,.0f} SAR")
    print(f"   MAE: {mae:,.0f} SAR")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Mean Price: {y.mean():,.0f} SAR")
    print(f"   Median Price: {y.median():,.0f} SAR")
    print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
    
    print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
    print("🏆 SAUDI HOUSING PREDICTION MODEL")
    print(f"   Accuracy: {r2*100:.2f}%")
    print(f"   Average Error: {mae:,.0f} SAR")
    print(f"   Percentage Error: {mape:.2f}%")
    print(f"   Samples Processed: {len(X):,}")
    
    # Feature importance if available
    if hasattr(model, 'feature_importances_'):
        print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

if __name__ == "__main__":
    main()