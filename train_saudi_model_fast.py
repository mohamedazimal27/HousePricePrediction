#!/usr/bin/env python3
"""
Fast training script for Saudi housing data with improved accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/processed/saudi_housing_english.csv')

print("=== FAST SAUDI HOUSING MODEL ===")
print(f"Dataset shape: {df.shape}")

# Enhanced feature engineering
print("\n=== FEATURE ENGINEERING ===")

# Create new features
df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['living_rooms']
df['luxury_score'] = (
    df['garage'] + df['driver_room'] + df['maid_room'] + 
    df['furnished'] + df['air_conditioning'] + df['pool'] + 
    df['front_yard'] + df['basement'] + df['duplex'] + 
    df['elevator'] + df['fireplace']
)
df['price_per_sqm'] = df['price'] / df['land_area']
df['age_category'] = pd.cut(df['property_age'], 
                            bins=[0, 5, 15, 30, 100], 
                            labels=['new', 'recent', 'old', 'very_old'])

# Encode categorical variables
le_city = LabelEncoder()
le_district = LabelEncoder()
le_front_direction = LabelEncoder()
le_age_category = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['city'].astype(str))
df['district_encoded'] = le_district.fit_transform(df['district'].astype(str))
df['front_direction_encoded'] = le_front_direction.fit_transform(df['front_direction'].astype(str))
df['age_category_encoded'] = le_age_category.fit_transform(df['age_category'].astype(str))

# Select features
features = [
    'city_encoded', 'district_encoded', 'front_direction_encoded',
    'land_area', 'property_age', 'bedrooms', 'bathrooms', 
    'living_rooms', 'garage', 'driver_room', 'maid_room',
    'furnished', 'air_conditioning', 'roof', 'pool', 
    'front_yard', 'basement', 'duplex', 'stairs', 
    'elevator', 'fireplace', 'total_rooms', 'luxury_score',
    'age_category_encoded'
]

# Filter available features
available_features = [f for f in features if f in df.columns]
print(f"Total features: {len(available_features)}")

# Prepare data
X = df[available_features].copy()
y = df['price']

# Handle missing values
X = X.fillna(X.median())
y = y.fillna(y.median())

# Remove outliers (properties with extreme prices)
Q1 = y.quantile(0.05)
Q3 = y.quantile(0.95)
mask = (y >= Q1) & (y <= Q3)
X = X[mask]
y = y[mask]

print(f"After outlier removal: {len(X)} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model comparison (faster version)
print("\n=== MODEL COMPARISON ===")

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    # Cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, 
                           cv=3, scoring='r2', n_jobs=-1)
    
    # Fit and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'cv_score': scores.mean(),
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }
    
    print(f"{name}:")
    print(f"  CV R²: {scores.mean():.4f}")
    print(f"  Test R²: {r2:.4f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  MAE: {mae:,.0f}")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
print(f"\nBest model: {best_model_name}")

# Final model with optimized parameters (pre-tuned)
print("\n=== FINAL MODEL ===")

if best_model_name == 'XGBoost':
    final_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
else:
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )

final_model.fit(X_train_scaled, y_train)
y_pred_final = final_model.predict(X_test_scaled)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_r2 = r2_score(y_test, y_pred_final)
final_mae = mean_absolute_error(y_test, y_pred_final)

print(f"\n=== FINAL MODEL PERFORMANCE ===")
print(f"R² Score: {final_r2:.4f}")
print(f"RMSE: {final_rmse:,.0f}")
print(f"MAE: {final_mae:,.0f}")
print(f"Accuracy improvement: {((final_r2 - 0.2836) / 0.2836 * 100):.1f}%")

# Create enhanced visualizations
print("\n=== CREATING ENHANCED VISUALIZATIONS ===")

# 1. Enhanced feature importance
plt.figure(figsize=(12, 8))
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title(f'Feature Importance - {best_model_name} (Enhanced Model)')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance_enhanced.png')
    plt.close()

# 2. Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_final, alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Price (SAR)')
plt.ylabel('Predicted Price (SAR)')
plt.title(f'Enhanced Model: Actual vs Predicted (R² = {final_r2:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted_enhanced.png')
plt.close()

# Save enhanced model
print("\n=== SAVING ENHANCED MODEL ===")
joblib.dump(final_model, 'enhanced_saudi_house_price_model.pkl')
joblib.dump(scaler, 'enhanced_model_scaler.pkl')
joblib.dump(available_features, 'enhanced_model_features.pkl')

# Save encoders
encoders = {
    'city_encoder': le_city,
    'district_encoder': le_district,
    'front_direction_encoder': le_front_direction,
    'age_category_encoder': le_age_category
}
joblib.dump(encoders, 'enhanced_model_encoders.pkl')

print("\n=== ENHANCED TRAINING COMPLETE ===")
print("Files created:")
print("- enhanced_saudi_house_price_model.pkl")
print("- enhanced_model_scaler.pkl")
print("- enhanced_model_features.pkl")
print("- enhanced_model_encoders.pkl")
print("- feature_importance_enhanced.png")
print("- actual_vs_predicted_enhanced.png")
