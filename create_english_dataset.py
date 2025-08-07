#!/usr/bin/env python3
"""
Create English version of Saudi housing dataset with manual translation
"""

import pandas as pd
import numpy as np

# Load the original Saudi dataset
df = pd.read_csv('SA_Aqar.csv')

# Define manual translations for column names
column_mapping = {
    'city': 'city',
    'district': 'district', 
    'front': 'front_direction',
    'size': 'land_area',
    'property_age': 'property_age',
    'bedrooms': 'bedrooms',
    'bathrooms': 'bathrooms',
    'livingrooms': 'living_rooms',
    'kitchen': 'kitchen',
    'garage': 'garage',
    'driver_room': 'driver_room',
    'maid_room': 'maid_room',
    'furnished': 'furnished',
    'ac': 'air_conditioning',
    'roof': 'roof',
    'pool': 'pool',
    'frontyard': 'front_yard',
    'basement': 'basement',
    'duplex': 'duplex',
    'stairs': 'stairs',
    'elevator': 'elevator',
    'fireplace': 'fireplace',
    'price': 'price'
}

# Define manual translations for categorical values
city_mapping = {
    ' الرياض': 'Riyadh',
    ' جدة': 'Jeddah', 
    ' الدمام': 'Dammam',
    ' الخبر': 'Khobar'
}

direction_mapping = {
    'شمال': 'North',
    'جنوب': 'South',
    'غرب': 'West',
    'شرق': 'East',
    'جنوب شرقي': 'Southeast',
    'جنوب غربي': 'Southwest',
    'شمال غربي': 'Northwest',
    'شمال شرقي': 'Northeast',
    '3 شوارع': '3 Streets',
    '4 شوارع': '4 Streets'
}

# Create a copy for English version
df_en = df.copy()

# Rename columns
df_en = df_en.rename(columns=column_mapping)

# Translate categorical values
df_en['city'] = df_en['city'].map(city_mapping)
df_en['front_direction'] = df_en['front_direction'].map(direction_mapping)

# Clean district names - remove Arabic "حي" (district) prefix and trim spaces
df_en['district'] = df_en['district'].str.replace('حي', '').str.strip()
df_en['district'] = df_en['district'].str.strip()

# Save the English version
df_en.to_csv('data/processed/saudi_housing_english.csv', index=False)

print("=== ENGLISH DATASET CREATED ===")
print(f"Shape: {df_en.shape}")
print(f"\nColumns: {list(df_en.columns)}")
print(f"\nCities: {df_en['city'].unique()}")
print(f"\nDirections: {df_en['front_direction'].unique()}")
print(f"\nSample districts: {df_en['district'].unique()[:10]}")
print("\nFirst few rows:")
print(df_en.head())

# Create summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(df_en.describe())
