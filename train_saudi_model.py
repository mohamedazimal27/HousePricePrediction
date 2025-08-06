#!/usr/bin/env python3
"""
Train XGBoost model on Saudi housing data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the English version of Saudi dataset
df = pd.read_csv('saudi_housing_english.csv')

print("=== SAUDI HOUSING DATASET ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Data Preprocessing
print("\n=== DATA PREPROCESSING ===")

# Select features for modeling based on available data
# We'll use the most relevant features for price prediction
features = [
    'bedrooms', 
    'bathrooms', 
    'land_area', 
    'property_age',
    'living_rooms',
    'garage',
    'driver_room',
    'maid_room',
    'furnished',
    'air_conditioning',
    'duplex'
]

# Check which features exist in the dataset
available_features = [f for f in features if f in df.columns]
print(f"Available features: {available_features}")

# Create feature matrix and target
X = df[available_features].copy()
y = df['price']

# Handle any missing values
X = X.fillna(0)
y = y.fillna(y.median())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Model Training
print("\n=== MODEL TRAINING ===")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Model Evaluation
print("\n=== MODEL EVALUATION ===")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Create visualizations
print("\n=== CREATING VISUALIZATIONS ===")

# 1. Feature Importance
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Saudi Housing Model')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_saudi.png')
plt.close()

# 2. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (SAR)')
plt.ylabel('Predicted Price (SAR)')
plt.title('Actual vs Predicted House Prices - Saudi Data')
plt.tight_layout()
plt.savefig('actual_vs_predicted_saudi.png')
plt.close()

# 3. Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of House Prices - Saudi Data')
plt.xlabel('Price (SAR)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('price_distribution_saudi.png')
plt.close()

# Save Model
print("\n=== SAVING MODEL ===")
joblib.dump(model, 'xgboost_saudi_house_price_model.pkl')
print("✅ Model saved as 'xgboost_saudi_house_price_model.pkl'")

# Save feature names for later use
feature_info = {
    'features': available_features,
    'feature_importance': dict(zip(available_features, model.feature_importances_))
}
joblib.dump(feature_info, 'saudi_model_features.pkl')

print("\n=== TRAINING COMPLETE ===")
print("Files created:")
print("- xgboost_saudi_house_price_model.pkl")
print("- saudi_model_features.pkl")
print("- feature_importance_saudi.png")
print("- actual_vs_predicted_saudi.png")
print("- price_distribution_saudi.png")
