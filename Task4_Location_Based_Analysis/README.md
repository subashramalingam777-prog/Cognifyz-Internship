# Task 4: Location-based Analysis

## Objective
Perform a geographical analysis of the restaurants in the dataset.

## Approach
1. Dropped rows with missing/zero latitude-longitude (placeholder entries).
2. Plotted all restaurants by lat/long to visualize geographic spread.
3. Grouped restaurants by city to find concentration and computed
   city-level stats: restaurant count, average rating, average cost for
   two, average price range.
4. Found the most common cuisine per top city.
5. Compared rating distributions across the 5 busiest cities.

## Key Insights
- **New Delhi** is by far the busiest city (5,240 restaurants), followed by
  Gurgaon and Noida — the dataset is heavily India-weighted (specifically
  the Delhi NCR region).
- Among cities with 20+ restaurants, **London** has the highest average
  rating.
- Among cities with 20+ restaurants, **Pune** has the highest average cost
  for two.
- **North Indian** is the most common cuisine across nearly all major
  Indian cities in the dataset; international cities skew toward
  **American**, **Continental**, or local cuisines.

## Files
- `task4_location_analysis.py` — full pipeline (run with `python3
  task4_location_analysis.py`, expects `Dataset_.csv` in the same folder)
- `restaurant_distribution_map.png` — scatter of all restaurants by
  lat/long
- `top_cities_restaurant_count.png` — top 15 cities by restaurant count
- `rating_distribution_top5_cities.png` — boxplot of ratings, top 5 cities
- `city_level_statistics.csv` — full per-city stats table
- `most_common_cuisine_by_city.csv`
- `insights.txt`
