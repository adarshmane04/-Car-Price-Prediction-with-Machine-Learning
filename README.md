# 🚗 Car Price Prediction with Machine Learning

A regression project predicting the **selling price of used cars** from features
such as brand, age, mileage, fuel type, and transmission — built end-to-end in a
single, Colab-ready Jupyter notebook.

## Dataset

**CAR DETAILS FROM CAR DEKHO** (used-car listings scraped from CarDekho.com),
the same dataset published on Kaggle at
[vehicle-dataset-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

| Column | Description |
|---|---|
| `name` | Car name (brand + model) |
| `year` | Manufacturing year |
| `selling_price` | Price of the car (INR) — **target variable** |
| `km_driven` | Total kilometers driven |
| `fuel` | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | First / Second / Third / Fourth & Above / Test Drive Car |

The notebook loads the dataset directly from a public raw-CSV URL, so it runs
with **zero manual downloads** — just open it in Colab and run all cells.

## How to run

1. Open [Google Colab](https://colab.research.google.com)
2. `File → Upload notebook` and select `Car_Price_Prediction.ipynb`
   (or drag it into Colab directly)
3. `Runtime → Run all`

No local paths, no Kaggle API key, no manual file uploads required.

## What the notebook covers

- [x] Data loading (direct from source, Colab-safe)
- [x] Data cleaning — nulls, duplicates, inconsistent categorical text (`"Petrol"` vs `"petrol"`)
- [x] Feature engineering — `car_age` from `year`, `brand` extracted from `name`
- [x] EDA — selling price distribution, price vs fuel type boxplot, price vs car age scatter plot, brand/transmission views
- [x] Encoding — ordinal encoding for `owner`, one-hot encoding for `fuel`/`seller_type`/`transmission`/`brand`
- [x] Feature correlation heatmap
- [x] Train/test split (80/20)
- [x] Three regression models — Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
- [x] Evaluation — MAE, RMSE, R² for each model, with comparison charts
- [x] Feature importance chart for the best-performing model
- [x] Actual vs. Predicted price plot

## Results (on this run)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | ~162,412 | ~378,000 | **0.556** |
| Linear Regression | ~186,138 | ~390,059 | 0.528 |
| Gradient Boosting | ~160,292 | ~392,609 | 0.521 |

Random Forest edged out as the best all-around performer on R² here, with
Gradient Boosting close behind on MAE. Exact numbers will vary slightly run to
run since the underlying data source may be updated.

The strongest predictors of price were consistently **car age, kilometers
driven, and brand** — matching real-world used-car pricing intuition.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## Possible extensions

- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Log-transform the target for better residual behavior
- Try XGBoost / LightGBM / CatBoost
- Deploy the trained model behind a simple Streamlit or Flask app
