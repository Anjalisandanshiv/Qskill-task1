import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# First, let's create a sample dataset (you can replace this with your own CSV file)
print("Creating sample sales data...")
np.random.seed(42)

# Generate sample e-commerce sales data
data = {
    'Product': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch'] * 100,
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 500),
    'Sales': np.random.randint(100, 2000, 500),
    'Quantity': np.random.randint(1, 50, 500),
    'Customer_Rating': np.random.uniform(1, 5, 500).round(1),
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 500)
}

df = pd.DataFrame(data)
df['Revenue'] = df['Sales'] * df['Quantity']

# Save to CSV
csv_filename = '/home/claude/sample_sales_data.csv'
df.to_csv(csv_filename, index=False)
print(f"Sample data saved to {csv_filename}\n")

# ============================================
# PART 1: LOAD AND EXPLORE THE DATA
# ============================================

print("="*60)
print("LOADING AND EXPLORING DATA")
print("="*60)

# Load the CSV file
df = pd.read_csv(csv_filename)

print("\n1. First 5 rows of the dataset:")
print(df.head())

print("\n2. Dataset Info:")
print(df.info())

print("\n3. Statistical Summary:")
print(df.describe())

print("\n4. Checking for missing values:")
print(df.isnull().sum())

# ============================================
# PART 2: BASIC DATA ANALYSIS
# ============================================

print("\n" + "="*60)
print("BASIC DATA ANALYSIS")
print("="*60)

# Calculate averages
print("\n1. Average Sales per Transaction:", df['Sales'].mean().round(2))
print("2. Average Quantity per Transaction:", df['Quantity'].mean().round(2))
print("3. Average Customer Rating:", df['Customer_Rating'].mean().round(2))
print("4. Average Revenue:", df['Revenue'].mean().round(2))

# Group analysis
print("\n5. Sales by Product:")
product_sales = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print(product_sales)

print("\n6. Sales by Region:")
region_sales = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print(region_sales)

print("\n7. Average Rating by Product:")
product_ratings = df.groupby('Product')['Customer_Rating'].mean().sort_values(ascending=False)
print(product_ratings)

# ============================================
# PART 3: CREATE VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. BAR CHART: Total Revenue by Product
plt.subplot(2, 3, 1)
product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
colors = plt.cm.Set3(range(len(product_revenue)))
product_revenue.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Total Revenue by Product', fontsize=14, fontweight='bold')
plt.xlabel('Product', fontsize=11)
plt.ylabel('Total Revenue ($)', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 2. BAR CHART: Average Customer Rating by Product
plt.subplot(2, 3, 2)
avg_ratings = df.groupby('Product')['Customer_Rating'].mean().sort_values(ascending=False)
avg_ratings.plot(kind='bar', color='skyblue', edgecolor='navy')
plt.title('Average Customer Rating by Product', fontsize=14, fontweight='bold')
plt.xlabel('Product', fontsize=11)
plt.ylabel('Average Rating', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 5)
plt.axhline(y=df['Customer_Rating'].mean(), color='red', linestyle='--', 
            label=f'Overall Avg: {df["Customer_Rating"].mean():.2f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 3. SCATTER PLOT: Sales vs Customer Rating
plt.subplot(2, 3, 3)
products = df['Product'].unique()
colors_scatter = plt.cm.Set2(range(len(products)))
for i, product in enumerate(products):
    product_data = df[df['Product'] == product]
    plt.scatter(product_data['Customer_Rating'], product_data['Sales'], 
               label=product, alpha=0.6, s=50, color=colors_scatter[i])
plt.title('Sales vs Customer Rating', fontsize=14, fontweight='bold')
plt.xlabel('Customer Rating', fontsize=11)
plt.ylabel('Sales ($)', fontsize=11)
plt.legend(loc='best', fontsize=8)
plt.grid(True, alpha=0.3)

# 4. HORIZONTAL BAR CHART: Revenue by Region
plt.subplot(2, 3, 4)
region_revenue = df.groupby('Region')['Revenue'].sum().sort_values()
region_revenue.plot(kind='barh', color='coral', edgecolor='darkred')
plt.title('Total Revenue by Region', fontsize=14, fontweight='bold')
plt.xlabel('Total Revenue ($)', fontsize=11)
plt.ylabel('Region', fontsize=11)
plt.grid(axis='x', alpha=0.3)

# 5. SCATTER PLOT: Quantity vs Revenue (with size representing rating)
plt.subplot(2, 3, 5)
plt.scatter(df['Quantity'], df['Revenue'], 
           c=df['Customer_Rating'], cmap='viridis', 
           s=df['Sales']/10, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.colorbar(label='Customer Rating')
plt.title('Quantity vs Revenue\n(Size = Sales, Color = Rating)', 
         fontsize=14, fontweight='bold')
plt.xlabel('Quantity', fontsize=11)
plt.ylabel('Revenue ($)', fontsize=11)
plt.grid(True, alpha=0.3)

# 6. PIE CHART: Market Share by Product
plt.subplot(2, 3, 6)
product_counts = df.groupby('Product')['Revenue'].sum()
plt.pie(product_counts, labels=product_counts.index, autopct='%1.1f%%',
       startangle=90, colors=colors)
plt.title('Revenue Distribution by Product', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Main visualizations saved to 'analysis_visualizations.png'")

# Create HEATMAP separately for better visibility
fig2, ax = plt.subplots(figsize=(10, 8))

# Create a pivot table for heatmap
heatmap_data = df.pivot_table(
    values='Revenue', 
    index='Product', 
    columns='Region', 
    aggfunc='sum'
)

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
           linewidths=0.5, cbar_kws={'label': 'Revenue ($)'})
plt.title('Revenue Heatmap: Product vs Region', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Product', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/revenue_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Heatmap saved to 'revenue_heatmap.png'")

# ============================================
# PART 4: INSIGHTS AND OBSERVATIONS
# ============================================

print("\n" + "="*60)
print("KEY INSIGHTS AND OBSERVATIONS")
print("="*60)

print("\n📊 REVENUE ANALYSIS:")
top_product = product_revenue.idxmax()
top_product_revenue = product_revenue.max()
print(f"   • {top_product} generates the highest revenue: ${top_product_revenue:,.2f}")
print(f"   • This represents {(top_product_revenue/product_revenue.sum()*100):.1f}% of total revenue")

print("\n⭐ CUSTOMER SATISFACTION:")
best_rated = avg_ratings.idxmax()
best_rating = avg_ratings.max()
print(f"   • {best_rated} has the highest average rating: {best_rating:.2f}/5.0")
lowest_rated = avg_ratings.idxmin()
lowest_rating = avg_ratings.min()
print(f"   • {lowest_rated} has the lowest average rating: {lowest_rating:.2f}/5.0")
print(f"   • Overall average rating across all products: {df['Customer_Rating'].mean():.2f}/5.0")

print("\n🌍 REGIONAL PERFORMANCE:")
top_region = region_sales.idxmax()
top_region_revenue = region_sales.max()
print(f"   • {top_region} region leads in sales: ${top_region_revenue:,.2f}")
print(f"   • Regional revenue variation: ${region_sales.max() - region_sales.min():,.2f}")

print("\n💰 TRANSACTION METRICS:")
print(f"   • Average transaction value: ${df['Sales'].mean():.2f}")
print(f"   • Average quantity per order: {df['Quantity'].mean():.1f} units")
print(f"   • Total transactions analyzed: {len(df):,}")
print(f"   • Total revenue: ${df['Revenue'].sum():,.2f}")

print("\n🔍 CORRELATION INSIGHTS:")
correlation = df['Sales'].corr(df['Customer_Rating'])
print(f"   • Correlation between Sales and Customer Rating: {correlation:.3f}")
if abs(correlation) > 0.3:
    print(f"   • There is a {'positive' if correlation > 0 else 'negative'} relationship")
else:
    print(f"   • There is a weak relationship between sales and ratings")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nFiles generated:")
print("  1. sample_sales_data.csv - Sample dataset")
print("  2. analysis_visualizations.png - Main charts")
print("  3. revenue_heatmap.png - Product vs Region heatmap")
