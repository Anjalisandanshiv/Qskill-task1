"""
House Price Prediction using Linear Regression
Complete pipeline: Data generation, preprocessing, training, and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("HOUSE PRICE PREDICTION - LINEAR REGRESSION MODEL")
print("=" * 70)

# ============================================================================
# STEP 1: Generate Realistic Housing Dataset
# ============================================================================
print("\n[STEP 1] Generating realistic housing dataset...")

n_samples = 1000

# Generate features
data = {
    'num_rooms': np.random.randint(1, 10, n_samples),
    'num_bedrooms': np.random.randint(1, 6, n_samples),
    'num_bathrooms': np.random.randint(1, 5, n_samples),
    'square_feet': np.random.randint(500, 5000, n_samples),
    'lot_size': np.random.randint(1000, 20000, n_samples),
    'year_built': np.random.randint(1950, 2024, n_samples),
    'garage_spaces': np.random.randint(0, 4, n_samples),
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
    'neighborhood_quality': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_samples),
    'has_pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'has_basement': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'crime_rate': np.random.uniform(0.01, 10, n_samples),
    'distance_to_city_center': np.random.uniform(1, 50, n_samples),
    'school_rating': np.random.uniform(1, 10, n_samples),
}

df = pd.DataFrame(data)

# Generate realistic house prices based on features
base_price = 100000

price = (
    base_price +
    df['num_rooms'] * 15000 +
    df['num_bedrooms'] * 20000 +
    df['num_bathrooms'] * 18000 +
    df['square_feet'] * 100 +
    df['lot_size'] * 5 +
    (2024 - df['year_built']) * (-2000) +  # Newer houses are more expensive
    df['garage_spaces'] * 12000 +
    df['has_pool'] * 50000 +
    df['has_basement'] * 30000 +
    df['crime_rate'] * (-5000) +
    df['distance_to_city_center'] * (-3000) +
    df['school_rating'] * 8000
)

# Add location premium
location_premium = df['location'].map({'Urban': 80000, 'Suburban': 40000, 'Rural': 0})
price += location_premium

# Add neighborhood quality premium
quality_premium = df['neighborhood_quality'].map({
    'Poor': -30000, 'Average': 0, 'Good': 40000, 'Excellent': 80000
})
price += quality_premium

# Add some random noise
price += np.random.normal(0, 30000, n_samples)

# Ensure no negative prices
df['price'] = np.maximum(price, 50000)

print(f"✓ Dataset created with {len(df)} samples and {len(df.columns)} features")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Save the dataset
df.to_csv('/home/claude/housing_data.csv', index=False)
print(f"\n✓ Dataset saved to 'housing_data.csv'")

# ============================================================================
# STEP 2: Exploratory Data Analysis
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 2] Exploratory Data Analysis")
print("=" * 70)

print("\nDataset Info:")
print(df.info())

print("\n\nStatistical Summary:")
print(df.describe())

print("\n\nMissing Values:")
print(df.isnull().sum())

print("\n\nPrice Statistics:")
print(f"Mean Price: ${df['price'].mean():,.2f}")
print(f"Median Price: ${df['price'].median():,.2f}")
print(f"Min Price: ${df['price'].min():,.2f}")
print(f"Max Price: ${df['price'].max():,.2f}")
print(f"Std Dev: ${df['price'].std():,.2f}")

# ============================================================================
# STEP 3: Data Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 3] Data Preprocessing")
print("=" * 70)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}

categorical_columns = ['location', 'neighborhood_quality']
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded '{col}': {list(le.classes_)}")

# Split the data
print("\nSplitting data into train and test sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

# Feature scaling
print("\nApplying Standard Scaling to features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Features scaled")

# ============================================================================
# STEP 4: Model Training
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 4] Training Linear Regression Model")
print("=" * 70)

# Train the model
model = LinearRegression()
print("\nTraining model...")
model.fit(X_train_scaled, y_train)
print("  ✓ Model trained successfully!")

# ============================================================================
# STEP 5: Model Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 5] Model Evaluation")
print("=" * 70)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  RMSE:      ${train_rmse:,.2f}")
print(f"  MAE:       ${train_mae:,.2f}")
print(f"  MSE:       ${train_mse:,.2f}")

print("\n📊 TEST SET PERFORMANCE:")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  RMSE:      ${test_rmse:,.2f}")
print(f"  MAE:       ${test_mae:,.2f}")
print(f"  MSE:       ${test_mse:,.2f}")

# Feature importance
print("\n📈 FEATURE IMPORTANCE (Coefficients):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False, key=abs)

for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:.<30} {row['Coefficient']:>15,.2f}")

print(f"\n  Intercept: {model.intercept_:,.2f}")

# ============================================================================
# STEP 6: Visualizations
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 6] Creating Visualizations")
print("=" * 70)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Actual vs Predicted (Test Set)
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Prices (Test Set)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='both')

# 2. Residuals Plot
ax2 = plt.subplot(2, 3, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
plt.ylabel('Residuals ($)', fontsize=12, fontweight='bold')
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='both')

# 3. Distribution of Residuals
ax3 = plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals ($)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(True, alpha=0.3)

# 4. Feature Importance
ax4 = plt.subplot(2, 3, 4)
importance_plot = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in importance_plot['Coefficient']]
plt.barh(importance_plot['Feature'], importance_plot['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Coefficients', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 5. Price Distribution
ax5 = plt.subplot(2, 3, 5)
plt.hist(df['price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Price ($)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='plain', axis='x')

# 6. Model Performance Comparison
ax6 = plt.subplot(2, 3, 6)
metrics = ['R²', 'RMSE/10000', 'MAE/10000']
train_values = [train_r2, train_rmse/10000, train_mae/10000]
test_values = [test_r2, test_rmse/10000, test_mae/10000]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
plt.bar(x + width/2, test_values, width, label='Testing', alpha=0.8)
plt.xlabel('Metrics', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/model_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to 'model_visualizations.png'")

# ============================================================================
# STEP 7: Sample Predictions
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 7] Sample Predictions")
print("=" * 70)

# Make predictions on first 5 test samples
print("\nPredictions on sample houses:")
print("-" * 70)

sample_predictions = pd.DataFrame({
    'Actual Price': y_test.head(5).values,
    'Predicted Price': y_test_pred[:5],
    'Difference': y_test.head(5).values - y_test_pred[:5],
    'Error %': ((y_test.head(5).values - y_test_pred[:5]) / y_test.head(5).values * 100)
})

for idx, row in sample_predictions.iterrows():
    print(f"\nHouse {idx + 1}:")
    print(f"  Actual:    ${row['Actual Price']:,.2f}")
    print(f"  Predicted: ${row['Predicted Price']:,.2f}")
    print(f"  Difference: ${row['Difference']:,.2f} ({row['Error %']:.2f}%)")

# ============================================================================
# STEP 8: Save Model and Results
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 8] Saving Model and Results")
print("=" * 70)

import pickle

# Save the model
with open('/home/claude/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved to 'house_price_model.pkl'")

# Save the scaler
with open('/home/claude/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved to 'scaler.pkl'")

# Save label encoders
with open('/home/claude/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Label encoders saved to 'label_encoders.pkl'")

# Save results to text file
with open('/home/claude/model_results.txt', 'w') as f:
    f.write("HOUSE PRICE PREDICTION - MODEL RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset Size: {len(df)} samples\n")
    f.write(f"Number of Features: {len(X.columns)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n\n")
    f.write("TEST SET PERFORMANCE:\n")
    f.write(f"  R² Score:  {test_r2:.4f}\n")
    f.write(f"  RMSE:      ${test_rmse:,.2f}\n")
    f.write(f"  MAE:       ${test_mae:,.2f}\n")
    f.write(f"  MSE:       ${test_mse:,.2f}\n\n")
    f.write("FEATURE COEFFICIENTS:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"  {row['Feature']:.<30} {row['Coefficient']:>15,.2f}\n")
    f.write(f"\n  Intercept: {model.intercept_:,.2f}\n")

print("✓ Results saved to 'model_results.txt'")

print("\n" + "=" * 70)
print("✅ HOUSE PRICE PREDICTION MODEL COMPLETE!")
print("=" * 70)
print("\nModel Performance Summary:")
print(f"  • R² Score: {test_r2:.4f} (explains {test_r2*100:.2f}% of variance)")
print(f"  • Average prediction error: ${test_mae:,.2f}")
print(f"  • Model can predict prices within ±${test_rmse:,.2f} on average")
print("\nFiles Generated:")
print("  1. housing_data.csv - Dataset")
print("  2. house_price_model.pkl - Trained model")
print("  3. scaler.pkl - Feature scaler")
print("  4. label_encoders.pkl - Categorical encoders")
print("  5. model_visualizations.png - Visualizations")
print("  6. model_results.txt - Detailed results")
print("=" * 70)
