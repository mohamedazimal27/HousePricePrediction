#!/usr/bin/env python3
"""
Improved training script for Saudi housing data with better data handling and feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the dataset with improved data quality checks"""
    print("1. Loading and cleaning dataset...")
    df = pd.read_csv('data/processed/saudi_housing_english.csv')
    print(f"   Initial dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Data quality improvements
    initial_count = len(df)
    
    # Remove extreme outliers more conservatively (1st to 99th percentile)
    price_lower = df['price'].quantile(0.01)
    price_upper = df['price'].quantile(0.99)
    df = df[(df['price'] >= price_lower) & (df['price'] <= price_upper)]
    
    # Remove properties with unrealistic land areas
    df = df[df['land_area'] >= 50]  # Minimum 50 sqm
    df = df[df['land_area'] <= 5000]  # Maximum 5000 sqm
    
    # Fix bedroom distribution bias - cap at reasonable maximum
    df = df[df['bedrooms'] <= 8]  # Cap bedrooms at 8
    
    print(f"   After cleaning: {len(df)} samples ({initial_count - len(df)} removed)")
    return df

def engineer_features(df):
    """Enhanced feature engineering with better domain knowledge"""
    print("2. Engineering enhanced features...")
    
    # Create meaningful derived features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['living_rooms']
    
    # Improved luxury score with weights
    luxury_features = ['garage', 'driver_room', 'maid_room', 'furnished', 
                      'air_conditioning', 'pool', 'front_yard', 'basement', 
                      'duplex', 'elevator', 'fireplace']
    df['luxury_score'] = df[luxury_features].sum(axis=1)
    
    # Price per square meter (important for real estate)
    df['price_per_sqm'] = df['price'] / df['land_area']
    
    # Room density (rooms per square meter)
    df['room_density'] = df['total_rooms'] / df['land_area'] * 1000
    
    # Property age categories
    df['age_category'] = pd.cut(df['property_age'], 
                               bins=[-1, 0, 5, 15, 50], 
                               labels=['new', 'recent', 'mature', 'old'])
    
    # Size categories
    df['size_category'] = pd.cut(df['land_area'], 
                                bins=[0, 200, 400, 800, 5000], 
                                labels=['small', 'medium', 'large', 'xlarge'])
    
    return df

def encode_categorical_features(df):
    """Improved categorical encoding with better handling"""
    print("3. Encoding categorical features...")
    
    encoders = {}
    categorical_cols = ['city', 'district', 'front_direction', 'age_category', 'size_category']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values properly for categorical columns
            if df[col].dtype.name == 'category':
                # For categorical columns, add 'Unknown' to categories first
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['Unknown'])
                df[col] = df[col].fillna('Unknown')
            else:
                # For object columns, simple fillna works
                df[col] = df[col].fillna('Unknown')
            
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders

def select_features(df):
    """Select the most relevant features based on domain knowledge"""
    features = [
        # Location features
        'city_encoded', 'district_encoded', 'front_direction_encoded',
        
        # Size features
        'land_area', 'bedrooms', 'bathrooms', 'living_rooms', 'total_rooms',
        'size_category_encoded', 'room_density',
        
        # Age and condition
        'property_age', 'age_category_encoded',
        
        # Amenities
        'garage', 'driver_room', 'maid_room', 'furnished', 
        'air_conditioning', 'roof', 'pool', 'front_yard', 
        'basement', 'duplex', 'stairs', 'elevator', 'fireplace',
        'luxury_score'
    ]
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    print(f"   Using {len(available_features)} features")
    
    return available_features

def train_improved_model(X_train, X_test, y_train, y_test):
    """Train model with better hyperparameters and validation"""
    print("4. Training improved XGBoost model...")
    
    # Use RobustScaler instead of StandardScaler for better outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Improved XGBoost parameters
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation for better model assessment
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"   Cross-validation R² scores: {cv_scores}")
    print(f"   Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, y_train):
    """Comprehensive model evaluation"""
    print("5. Evaluating model performance...")
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Additional metrics
    mean_price = y_test.mean()
    median_price = y_test.median()
    
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
    print(f"   RMSE: {rmse:,.0f} SAR")
    print(f"   MAE: {mae:,.0f} SAR")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Mean Price: {mean_price:,.0f} SAR")
    print(f"   Median Price: {median_price:,.0f} SAR")
    print(f"   Error Rate: {(rmse/mean_price*100):.2f}%")
    
    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
        'mean_price': mean_price, 'median_price': median_price
    }

def create_visualizations(model, features, df, metrics):
    """Create improved visualizations"""
    print("6. Creating enhanced visualizations...")
    
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # 1. Feature Importance with better formatting
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        top_features = feature_importance.head(20)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 20 Feature Importance - Improved Saudi Housing Model')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Price distribution by bedroom count
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='bedrooms', y='price')
    plt.title('Price Distribution by Bedroom Count')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price (SAR)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/price_by_bedrooms_improved.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Enhanced visualizations saved")

def main():
    """Main training pipeline with improvements"""
    print("=== IMPROVED SAUDI HOUSING PRICE PREDICTION MODEL ===")
    print("Enhanced with better data cleaning and feature engineering")
    print()
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Feature engineering
    df = engineer_features(df)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Select features
    features = select_features(df)
    
    # Prepare data
    X = df[features].copy()
    y = df['price']
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    print(f"   Final dataset: {len(X)} samples, {len(features)} features")
    
    # Split data with stratification by price quartiles
    price_quartiles = pd.qcut(y, q=4, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=price_quartiles
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    model, scaler = train_improved_model(X_train, X_test, y_train, y_test)
    
    # Evaluate model
    metrics = evaluate_model(model, scaler, X_test, y_test, y_train)
    
    # Create visualizations
    create_visualizations(model, features, df, metrics)
    
    # Save improved model
    print("\n7. Saving improved model components...")
    joblib.dump(model, 'models/saved/improved_model.pkl')
    joblib.dump(scaler, 'models/saved/improved_scaler.pkl')
    joblib.dump(features, 'models/saved/improved_features.pkl')
    joblib.dump(encoders, 'models/saved/improved_encoders.pkl')
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Improved model saved with {metrics['r2']*100:.2f}% accuracy!")
    print()
    print("Improvements made:")
    print("- Better outlier handling with robust scaling")
    print("- Enhanced feature engineering with domain knowledge")
    print("- Cross-validation for model assessment")
    print("- Stratified train-test split")
    print("- More conservative data cleaning")
    print("- Additional evaluation metrics")

if __name__ == "__main__":
    main()