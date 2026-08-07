"""
Cognifyz ML Internship - Task 4
Location-based Analysis
--------------------------------
Objective: Perform a geographical analysis of the restaurants in the dataset.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset_.csv"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# Drop rows with missing/zero coordinates (a handful of placeholder entries)
df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()

# ---------------------------------------------------------------
# 2. Visualize distribution of restaurants on a map (scatter)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["Longitude"], df["Latitude"], s=4, alpha=0.4, c="darkorange")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Distribution of Restaurants")
plt.tight_layout()
plt.savefig(BASE_DIR / "restaurant_distribution_map.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Group by city: concentration of restaurants
# ---------------------------------------------------------------
city_counts = df["City"].value_counts().head(15)
plt.figure(figsize=(10, 6))
city_counts.plot(kind="bar", color="steelblue")
plt.title("Top 15 Cities by Number of Restaurants")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(BASE_DIR / "top_cities_restaurant_count.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Stats by city: avg rating, avg cost, top cuisines
# ---------------------------------------------------------------
city_stats = df.groupby("City").agg(
    Restaurant_Count=("Restaurant ID", "count"),
    Avg_Rating=("Aggregate rating", "mean"),
    Avg_Cost_For_Two=("Average Cost for two", "mean"),
    Avg_Price_Range=("Price range", "mean"),
).sort_values("Restaurant_Count", ascending=False)

city_stats.to_csv(BASE_DIR / "city_level_statistics.csv")
print("\nTop 10 cities by restaurant count, with stats:")
print(city_stats.head(10))

# Most common cuisine per top city
top_15_cities = city_counts.index
cuisine_by_city = {}
for city in top_15_cities:
    subset = df[df["City"] == city]
    cuisines_series = subset["Cuisines"].dropna().str.split(",").explode().str.strip()
    if len(cuisines_series) > 0:
        cuisine_by_city[city] = cuisines_series.value_counts().idxmax()
cuisine_by_city_df = pd.Series(cuisine_by_city, name="Most Common Cuisine")
cuisine_by_city_df.to_csv(BASE_DIR / "most_common_cuisine_by_city.csv")
print("\nMost common cuisine per top city:")
print(cuisine_by_city_df)

# ---------------------------------------------------------------
# 5. Rating distribution across top cities (boxplot)
# ---------------------------------------------------------------
top5_cities = city_counts.head(5).index
plt.figure(figsize=(9, 6))
sns.boxplot(data=df[df["City"].isin(top5_cities)], x="City", y="Aggregate rating")
plt.title("Rating Distribution in Top 5 Cities (by restaurant count)")
plt.tight_layout()
plt.savefig(BASE_DIR / "rating_distribution_top5_cities.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Key insights (auto-derived, printed + saved)
# ---------------------------------------------------------------
insights = []
busiest_city = city_stats.index[0]
insights.append(f"Busiest city by restaurant count: {busiest_city} ({int(city_stats.iloc[0]['Restaurant_Count'])} restaurants).")

highest_rated_city = city_stats[city_stats["Restaurant_Count"] >= 20].sort_values("Avg_Rating", ascending=False).index[0]
insights.append(f"Among cities with 20+ restaurants, highest average rating: {highest_rated_city}.")

most_expensive_city = city_stats[city_stats["Restaurant_Count"] >= 20].sort_values("Avg_Cost_For_Two", ascending=False).index[0]
insights.append(f"Among cities with 20+ restaurants, highest average cost for two: {most_expensive_city}.")

with open(BASE_DIR / "insights.txt", "w") as f:
    for line in insights:
        print(line)
        f.write(line + "\n")

print("\nDone. See city_level_statistics.csv, most_common_cuisine_by_city.csv, insights.txt, and PNG charts.")
