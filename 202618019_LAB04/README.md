# DS605 Lab Assignment 4
## Airbnb Price Prediction

### Student Information

- **Student ID:** 202618019
- **Name:** Jayesh Hedaoo
- **Course:** DS605 - Fundamentals of Machine Learning

---

## Project Overview

This project develops an end-to-end machine learning system for
predicting the nightly price of Airbnb listings in New York City.

The project includes data exploration, preprocessing, feature
engineering, regression model comparison, hyperparameter tuning,
model evaluation, and deployment using Streamlit.

---

## Dataset

The project uses the New York City Airbnb Open Data dataset:

`AB_NYC_2019.csv`

The target variable is:

`price`

---

## Machine Learning Workflow

1. Exploratory Data Analysis
2. Missing value analysis
3. Duplicate detection
4. Data cleaning
5. Outlier treatment
6. Feature engineering
7. Feature selection
8. Train-test split
9. Data preprocessing
10. Regression model training
11. Model comparison
12. Hyperparameter tuning
13. Final model evaluation
14. Model serialization
15. Streamlit deployment

---

## Models Used

The following regression models were compared:

- Ridge Regression
- Random Forest Regression
- HistGradientBoosting Regression

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Feature Engineering

The following derived features were created:

- `availability_ratio`
- `reviews_log`
- `minimum_nights_log`

---

## Final Model

The final trained model and preprocessing pipeline are saved as:

`airbnb_price_model.pkl`

The saved pipeline is loaded by the Streamlit application.

---

## Streamlit Application

The application accepts Airbnb listing information and predicts
the estimated nightly price.

Run the application using:

```bash
streamlit run app.py