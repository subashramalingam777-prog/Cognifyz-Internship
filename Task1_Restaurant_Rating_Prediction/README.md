# Task 1: Predict Restaurant Ratings

## Objective
Build a machine learning model to predict a restaurant's aggregate rating
based on other features in the dataset.

## Approach
1. **Data cleaning:** Removed restaurants with 0 votes (rating = 0 / "Not
   rated" — these aren't genuine ratings and would bias the model).
2. **Preprocessing:** Filled missing `Cuisines` values, label-encoded
   categorical columns (`Currency`, `City`), and mapped Yes/No columns to
   1/0.
3. **Models trained:**
   - Linear Regression (baseline)
   - Decision Tree Regression
4. **Evaluation metrics:** Mean Squared Error (MSE) and R².

## Results
| Model | MSE | R² |
|---|---|---|
| Linear Regression | 1.12 | 0.26 |
| Decision Tree Regression | 0.12 | 0.92 |

The Decision Tree performs far better — ratings turn out to be strongly
driven by **number of Votes** (a restaurant with more votes tends to have a
more "settled," typically higher, rating), followed by Country Code and
City. See `feature_importance.png`.

## Files
- `task1_rating_prediction.py` — full pipeline (run with `python3
  task1_rating_prediction.py`, expects `Dataset_.csv` in the same folder)
- `results.txt` — metrics
- `feature_importance.png` / `feature_importance.csv`
- `actual_vs_predicted.png` — Decision Tree predictions vs actual ratings
