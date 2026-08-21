# DS605: Fundamentals of Machine Learning
## Lab Assignment 3 - Scikit-learn Data Preprocessing and Model Performance Evaluation

**Name:** Jayesh Hedaoo  
**ID:** 202618019

### Dataset

Hotel Booking Demand dataset.

### Objective

Build and compare Scikit-learn preprocessing pipelines and evaluate two classification models.

### Preprocessing

Two preprocessing pipelines were implemented:

- Pipeline A: KNN Imputer + StandardScaler
- Pipeline B: KNN Imputer + MinMaxScaler
- Categorical features: Most Frequent Imputation + OneHotEncoder

### Models

- Logistic Regression
- Decision Tree Classifier

### Results

| Model | Testing Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression + Pipeline A | 0.8166 | 0.8081 | 0.6620 | 0.7278 |
| Logistic Regression + Pipeline B | 0.8115 | 0.8021 | 0.6520 | 0.7193 |
| Decision Tree + Pipeline A | 0.8563 | 0.8022 | 0.8123 | 0.8072 |
| Decision Tree + Pipeline B | 0.8561 | 0.8017 | 0.8124 | 0.8070 |

### Final Observations

1. Decision Tree with Pipeline A gives the best overall result.
2. StandardScaler performs slightly better than MinMaxScaler for Logistic Regression.
3. Scaling has very little effect on the Decision Tree.
4. The Decision Tree shows possible overfitting because its training accuracy is 0.9962 while testing accuracy is approximately 0.856.
5. Decision Tree with Pipeline A has the highest testing accuracy and F1-score.

### Files

- `LAB3.ipynb` - Complete Jupyter Notebook
- `hotel_bookings_cleaned.csv` - Cleaned dataset
- `model_comparison.csv` - Model comparison results
- `figures/` - Confusion matrix figures