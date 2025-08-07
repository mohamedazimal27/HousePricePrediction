#!/usr/bin/env python3
"""
Check the accuracy of the presentation model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== PRESENTATION MODEL ACCURACY CHECK ===")
    
    # Load the presentation model components
    try:
        model = joblib.load('models/saved/presentation_model.pkl')
        scaler = joblib.load('models/saved/presentation_scaler.pkl')
        features = joblib.load('models/saved/presentation_features.pkl')
        encoders = joblib.load('models/saved/presentation_encoders.pkl')
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
    
    # Create new features (same as training)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['living_rooms']
    df['luxury_score'] = (
        df['garage'] + df['driver_room'] + df['maid_room'] + 
        df['furnished'] + df['air_conditioning'] + df['pool'] + 
        df['front_yard'] + df['basement'] + df['duplex'] + 
        df['elevator'] + df['fireplace']
    )
    df['price_per_sqm'] = df['price'] / df['land_area']
    
    # Apply categorical encoding (same as training)
    categorical_cols = ['city', 'district', 'front_direction']
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
    
    # Remove outliers (same as training)
    Q1 = y.quantile(0.05)
    Q3 = y.quantile(0.95)
    mask = (y >= Q1) & (y <= Q3)
    X = X[mask]
    y = y[mask]
    
    print(f"   Using {len(available_features)} features")
    print(f"   Samples after preprocessing: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
    print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
    
    print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
    print("🏆 PRESENTATION MODEL")
    print(f"   Accuracy: {r2*100:.2f}%")
    print(f"   Average Error: {mae:,.0f} SAR")
    print(f"   Error Rate: {(rmse/y.mean()*100):.2f}%")
    
    # Compare with original model
    print(f"\n=== COMPARISON WITH ORIGINAL MODEL ===")
    print("Original Model (Baseline):")
    print("   Accuracy: 28.36%")
    print("   Average Error: 19,735 SAR")
    print("   Error Rate: 59.67%")
    print("\nPresentation Model (Improved):")
    print(f"   Accuracy: {r2*100:.2f}% (+{((r2 - 0.2836) / 0.2836 * 100):.0f}% improvement)")
    print(f"   Average Error: {mae:,.0f} SAR ({((19735 - mae) / 19735 * 100):.0f}% reduction)")
    print(f"   Error Rate: {(rmse/y.mean()*100):.2f}% ({((59.67 - (rmse/y.mean()*100)) / 59.67 * 100):.0f}% improvement)")

if __name__ == "__main__":
    main()
