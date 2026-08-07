# Task 3: Cuisine Classification

## Objective
Develop a machine learning model to classify restaurants based on their
cuisines.

## Approach
1. Since many restaurants list multiple cuisines (e.g. "French, Japanese,
   Desserts"), the **primary (first-listed) cuisine** was used as the
   classification target.
2. Restricted to the **top 10 most common cuisines** — the full dataset has
   1,800+ unique cuisine combinations, most appearing only once, which
   isn't learnable.
3. Features used: Country Code, City, Average Cost for two, Price range,
   Votes, Aggregate rating, Has Table booking, Has Online delivery.
4. **Models trained:**
   - Logistic Regression
   - Random Forest Classifier
5. **Evaluation metrics:** Accuracy, weighted Precision, weighted Recall,
   and a per-class classification report.

## Results
| Model | Accuracy | Precision (weighted) | Recall (weighted) |
|---|---|---|---|
| Logistic Regression | 0.45 | 0.30 | 0.45 |
| Random Forest | 0.43 | 0.38 | 0.43 |

## Analysis
- **North Indian** and **American** cuisines classify best (72% and 57%
  recall) — likely because they're the majority class and/or have distinct
  cost/city patterns.
- **South Indian** and **Chinese** are hardest to distinguish (under 15%
  accuracy) — these cuisines overlap heavily in price range and city with
  other categories, so cost/location features alone don't separate them
  well. Text-based features from the cuisine description itself would help
  more than tabular features here.
- See `confusion_matrix.png` for a full breakdown of where the model
  confuses cuisines, and `per_cuisine_accuracy.csv` for per-class numbers.

## Files
- `task3_cuisine_classification.py` — full pipeline (run with `python3
  task3_cuisine_classification.py`, expects `Dataset_.csv` in the same
  folder)
- `results.txt` — metrics + classification report
- `confusion_matrix.png`
- `per_cuisine_accuracy.csv`
