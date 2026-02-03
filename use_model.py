"""
Using the Trained House Price Prediction Model
This script demonstrates how to load and use the trained model for predictions
"""

import pickle
import pandas as pd
import numpy as np

print("=" * 70)
print("USING THE TRAINED HOUSE PRICE PREDICTION MODEL")
print("=" * 70)

# Load the trained model, scaler, and encoders
print("\nLoading trained model and preprocessing objects...")
with open('/home/claude/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/home/claude/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/home/claude/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print("✓ Model loaded successfully!")

# Function to predict house price
def predict_house_price(house_features):
    """
    Predict house price given features
    
    Parameters:
    -----------
    house_features : dict
        Dictionary containing house features
    
    Returns:
    --------
    float : Predicted price
    """
    # Create a DataFrame from the input
    df = pd.DataFrame([house_features])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Scale the features
    df_scaled = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    
    return prediction

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE PREDICTIONS")
print("=" * 70)

# Example 1: Luxury Urban House
print("\n📍 Example 1: Luxury Urban House")
print("-" * 70)
house1 = {
    'num_rooms': 8,
    'num_bedrooms': 5,
    'num_bathrooms': 4,
    'square_feet': 4500,
    'lot_size': 15000,
    'year_built': 2020,
    'garage_spaces': 3,
    'location': 'Urban',
    'neighborhood_quality': 'Excellent',
    'has_pool': 1,
    'has_basement': 1,
    'crime_rate': 0.5,
    'distance_to_city_center': 2.0,
    'school_rating': 9.5
}

print("Features:")
for key, value in house1.items():
    print(f"  {key:.<30} {value}")

predicted_price1 = predict_house_price(house1)
print(f"\n💰 Predicted Price: ${predicted_price1:,.2f}")

# Example 2: Modest Suburban House
print("\n\n📍 Example 2: Modest Suburban House")
print("-" * 70)
house2 = {
    'num_rooms': 5,
    'num_bedrooms': 3,
    'num_bathrooms': 2,
    'square_feet': 1800,
    'lot_size': 5000,
    'year_built': 2000,
    'garage_spaces': 2,
    'location': 'Suburban',
    'neighborhood_quality': 'Good',
    'has_pool': 0,
    'has_basement': 1,
    'crime_rate': 2.0,
    'distance_to_city_center': 15.0,
    'school_rating': 7.5
}

print("Features:")
for key, value in house2.items():
    print(f"  {key:.<30} {value}")

predicted_price2 = predict_house_price(house2)
print(f"\n💰 Predicted Price: ${predicted_price2:,.2f}")

# Example 3: Budget Rural House
print("\n\n📍 Example 3: Budget Rural House")
print("-" * 70)
house3 = {
    'num_rooms': 4,
    'num_bedrooms': 2,
    'num_bathrooms': 1,
    'square_feet': 1200,
    'lot_size': 8000,
    'year_built': 1980,
    'garage_spaces': 1,
    'location': 'Rural',
    'neighborhood_quality': 'Average',
    'has_pool': 0,
    'has_basement': 0,
    'crime_rate': 1.5,
    'distance_to_city_center': 40.0,
    'school_rating': 6.0
}

print("Features:")
for key, value in house3.items():
    print(f"  {key:.<30} {value}")

predicted_price3 = predict_house_price(house3)
print(f"\n💰 Predicted Price: ${predicted_price3:,.2f}")

# ============================================================================
# BATCH PREDICTIONS
# ============================================================================
print("\n\n" + "=" * 70)
print("BATCH PREDICTIONS")
print("=" * 70)

# Create multiple houses for batch prediction
batch_houses = [
    {'num_rooms': 6, 'num_bedrooms': 3, 'num_bathrooms': 2, 'square_feet': 2500,
     'lot_size': 7000, 'year_built': 2015, 'garage_spaces': 2, 'location': 'Urban',
     'neighborhood_quality': 'Good', 'has_pool': 1, 'has_basement': 1,
     'crime_rate': 1.0, 'distance_to_city_center': 5.0, 'school_rating': 8.5},
    
    {'num_rooms': 5, 'num_bedrooms': 2, 'num_bathrooms': 2, 'square_feet': 1600,
     'lot_size': 4000, 'year_built': 2010, 'garage_spaces': 1, 'location': 'Suburban',
     'neighborhood_quality': 'Average', 'has_pool': 0, 'has_basement': 0,
     'crime_rate': 3.0, 'distance_to_city_center': 20.0, 'school_rating': 6.5},
    
    {'num_rooms': 7, 'num_bedrooms': 4, 'num_bathrooms': 3, 'square_feet': 3500,
     'lot_size': 12000, 'year_built': 2022, 'garage_spaces': 3, 'location': 'Urban',
     'neighborhood_quality': 'Excellent', 'has_pool': 1, 'has_basement': 1,
     'crime_rate': 0.2, 'distance_to_city_center': 1.5, 'school_rating': 9.8}
]

print("\nPredicting prices for 3 houses:")
print("-" * 70)
for i, house in enumerate(batch_houses, 1):
    price = predict_house_price(house)
    print(f"House {i}: ${price:,.2f} | "
          f"{house['num_bedrooms']} bed, {house['num_bathrooms']} bath, "
          f"{house['square_feet']} sqft, {house['location']}")

# ============================================================================
# FEATURE SENSITIVITY ANALYSIS
# ============================================================================
print("\n\n" + "=" * 70)
print("FEATURE SENSITIVITY ANALYSIS")
print("=" * 70)

base_house = {
    'num_rooms': 5,
    'num_bedrooms': 3,
    'num_bathrooms': 2,
    'square_feet': 2000,
    'lot_size': 6000,
    'year_built': 2010,
    'garage_spaces': 2,
    'location': 'Suburban',
    'neighborhood_quality': 'Good',
    'has_pool': 0,
    'has_basement': 1,
    'crime_rate': 2.0,
    'distance_to_city_center': 15.0,
    'school_rating': 7.0
}

base_price = predict_house_price(base_house)
print(f"\nBase House Price: ${base_price:,.2f}")
print("\nImpact of changing individual features:")
print("-" * 70)

# Test impact of adding 1000 square feet
modified = base_house.copy()
modified['square_feet'] = 3000
new_price = predict_house_price(modified)
print(f"Adding 1000 sqft:           ${new_price - base_price:>+15,.2f}")

# Test impact of adding a bedroom
modified = base_house.copy()
modified['num_bedrooms'] = 4
new_price = predict_house_price(modified)
print(f"Adding 1 bedroom:           ${new_price - base_price:>+15,.2f}")

# Test impact of adding a bathroom
modified = base_house.copy()
modified['num_bathrooms'] = 3
new_price = predict_house_price(modified)
print(f"Adding 1 bathroom:          ${new_price - base_price:>+15,.2f}")

# Test impact of adding a pool
modified = base_house.copy()
modified['has_pool'] = 1
new_price = predict_house_price(modified)
print(f"Adding a pool:              ${new_price - base_price:>+15,.2f}")

# Test impact of upgrading location to Urban
modified = base_house.copy()
modified['location'] = 'Urban'
new_price = predict_house_price(modified)
print(f"Upgrading to Urban:         ${new_price - base_price:>+15,.2f}")

# Test impact of newer construction
modified = base_house.copy()
modified['year_built'] = 2020
new_price = predict_house_price(modified)
print(f"Built in 2020 (vs 2010):    ${new_price - base_price:>+15,.2f}")

# Test impact of better school rating
modified = base_house.copy()
modified['school_rating'] = 9.0
new_price = predict_house_price(modified)
print(f"Better schools (9.0 vs 7.0):${new_price - base_price:>+15,.2f}")

# Test impact of lower crime rate
modified = base_house.copy()
modified['crime_rate'] = 0.5
new_price = predict_house_price(modified)
print(f"Lower crime (0.5 vs 2.0):   ${new_price - base_price:>+15,.2f}")

print("\n" + "=" * 70)
print("✅ PREDICTION EXAMPLES COMPLETE!")
print("=" * 70)
print("\nYou can now use this model to predict house prices!")
print("Simply create a dictionary with house features and call predict_house_price()")
