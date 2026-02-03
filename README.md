# House Price Prediction using Linear Regression

## Project Overview
This project implements a complete machine learning pipeline for predicting house prices based on various property features including location, size, age, amenities, and neighborhood characteristics.

## Model Performance

### Test Set Metrics:
- **R² Score**: 0.9039 (90.39% variance explained)
- **RMSE**: $52,433.37
- **MAE**: $44,263.89
- **MSE**: $2,749,258,320.77

The model successfully explains over 90% of the variance in house prices, with an average prediction error of approximately $44,000.

## Dataset

### Features (14 input variables):
1. **num_rooms**: Total number of rooms in the house (1-9)
2. **num_bedrooms**: Number of bedrooms (1-5)
3. **num_bathrooms**: Number of bathrooms (1-4)
4. **square_feet**: Total living area in square feet (500-5000)
5. **lot_size**: Property lot size in square feet (1000-20000)
6. **year_built**: Year the house was constructed (1950-2023)
7. **garage_spaces**: Number of garage parking spaces (0-3)
8. **location**: Geographic location (Urban/Suburban/Rural)
9. **neighborhood_quality**: Quality rating (Poor/Average/Good/Excellent)
10. **has_pool**: Pool presence (0=No, 1=Yes)
11. **has_basement**: Basement presence (0=No, 1=Yes)
12. **crime_rate**: Local crime rate index (0.01-10)
13. **distance_to_city_center**: Distance to downtown in miles (1-50)
14. **school_rating**: Local school quality rating (1-10)

### Target Variable:
- **price**: House sale price in USD ($64,974 - $1,057,210)

### Dataset Statistics:
- **Total samples**: 1,000 houses
- **Training set**: 800 samples (80%)
- **Test set**: 200 samples (20%)
- **Mean price**: $594,797
- **Median price**: $595,699

## Feature Importance (Regression Coefficients)

The most influential features on house price (in order of impact):

1. **square_feet**: +$125,393 (Most important)
2. **distance_to_city_center**: -$42,480 (Negative impact)
3. **year_built**: +$40,950 (Newer = higher price)
4. **num_rooms**: +$37,838
5. **num_bedrooms**: +$29,372
6. **location**: +$29,086
7. **lot_size**: +$27,944
8. **has_pool**: +$25,666
9. **num_bathrooms**: +$20,971
10. **school_rating**: +$17,977

## Files Included

### 1. Main Scripts:
- **house_price_prediction.py**: Complete pipeline (data generation, preprocessing, training, evaluation)
- **use_model.py**: Demonstrates how to use the trained model for predictions

### 2. Data Files:
- **housing_data.csv**: Full dataset (1000 samples)

### 3. Model Files:
- **house_price_model.pkl**: Trained Linear Regression model
- **scaler.pkl**: StandardScaler for feature normalization
- **label_encoders.pkl**: Encoders for categorical variables

### 4. Results & Visualizations:
- **model_visualizations.png**: Comprehensive visualization dashboard
- **model_results.txt**: Detailed performance metrics and coefficients

## How to Use the Model

### Loading the Model:
```python
import pickle
import pandas as pd

# Load the model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
```

### Making Predictions:
```python
# Create house features
house = {
    'num_rooms': 6,
    'num_bedrooms': 3,
    'num_bathrooms': 2,
    'square_feet': 2500,
    'lot_size': 7000,
    'year_built': 2015,
    'garage_spaces': 2,
    'location': 'Urban',
    'neighborhood_quality': 'Good',
    'has_pool': 1,
    'has_basement': 1,
    'crime_rate': 1.0,
    'distance_to_city_center': 5.0,
    'school_rating': 8.5
}

# Convert to DataFrame
df = pd.DataFrame([house])

# Encode categorical variables
for col, encoder in label_encoders.items():
    df[col] = encoder.transform(df[col])

# Scale features
df_scaled = scaler.transform(df)

# Predict
predicted_price = model.predict(df_scaled)[0]
print(f"Predicted Price: ${predicted_price:,.2f}")
```

## Example Predictions

### Luxury Urban House:
- Features: 5 bed, 4 bath, 4500 sqft, pool, excellent neighborhood
- **Predicted Price: $1,199,091**

### Modest Suburban House:
- Features: 3 bed, 2 bath, 1800 sqft, good neighborhood
- **Predicted Price: $542,496**

### Budget Rural House:
- Features: 2 bed, 1 bath, 1200 sqft, average neighborhood
- **Predicted Price: $260,999**

## Feature Impact Analysis

Impact of individual feature changes on base house ($583,511):

- Adding 1000 sqft: **+$99,260**
- Adding pool: **+$55,136**
- Upgrading to Urban location: **+$36,251**
- Adding 1 bedroom: **+$20,715**
- Built 10 years newer: **+$19,588**
- Adding 1 bathroom: **+$18,815**
- Better schools (+2 points): **+$13,845**
- Lower crime (-1.5 points): **+$8,162**

## Model Methodology

### 1. Data Preprocessing:
- Encoded categorical variables (location, neighborhood quality)
- Applied standard scaling to all features
- Split data into 80% training, 20% testing

### 2. Model Training:
- Algorithm: Linear Regression (Ordinary Least Squares)
- No regularization applied
- Fitted on scaled training data

### 3. Evaluation:
- Assessed using R², RMSE, MAE, MSE
- Validated on held-out test set
- Analyzed residuals for model assumptions

## Visualizations

The `model_visualizations.png` file contains:
1. Actual vs Predicted prices scatter plot
2. Residual distribution plot
3. Feature importance bar chart
4. Price distribution histogram
5. Model performance comparison
6. Residual vs predicted plot

## Key Insights

1. **Square footage is king**: The single most important factor, adding ~$100/sqft
2. **Location matters**: Urban properties command significant premium
3. **Proximity premium**: Closer to city center adds substantial value
4. **Modern construction**: Newer homes are valued significantly higher
5. **Amenities pay off**: Pools and basements add considerable value
6. **Safety first**: Lower crime rates positively impact pricing
7. **Education matters**: Better school ratings increase home values

## Limitations & Future Improvements

### Current Limitations:
- Synthetic dataset (not real Kaggle data due to network restrictions)
- Linear model assumes linear relationships
- No interaction terms between features
- Doesn't account for market trends or seasonality

### Potential Improvements:
1. Use real-world data from Kaggle or Zillow
2. Try polynomial features for non-linear relationships
3. Experiment with Ridge/Lasso regression for feature selection
4. Add interaction terms (e.g., location × square_feet)
5. Incorporate time-series data for market trends
6. Try ensemble methods (Random Forest, Gradient Boosting)
7. Include additional features (walkability score, recent renovations)

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Running the Project

### Train the model:
```bash
python house_price_prediction.py
```

### Use the model:
```bash
python use_model.py
```

## License
This is an educational project for demonstrating machine learning concepts.

## Author
Created as a demonstration of end-to-end machine learning pipeline for house price prediction.

---

**Last Updated**: February 2026
