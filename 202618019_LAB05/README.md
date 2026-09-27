# DS605 Lab 5 - Machine Learning with Scikit-learn and From Scratch

## Overview

This lab implements regression and classification models using the UCI
Productivity Prediction of Garment Employees dataset.

The same machine-learning workflow is implemented in two ways:

1. Using Scikit-learn
2. From scratch using NumPy and Pandas

The predictive performance and execution time of the implementations are
compared, followed by optimization of the manual Logistic Regression model.

## Dataset

Dataset:
UCI Productivity Prediction of Garment Employees

File:

data/garments_worker_productivity.csv

The dataset contains production-related features such as department,
team, overtime, incentives, work in progress, number of workers,
targeted productivity, and actual productivity.

## Targets

### Regression

Target:

actual_productivity

Model:

Linear Regression

### Classification

A new target named `MeetsTarget` is created:

- 1 if actual_productivity >= targeted_productivity
- 0 otherwise

`actual_productivity` is not used as an input feature for classification.

Model:

Logistic Regression

## Part A - Scikit-learn

The Scikit-learn implementation performs:

- Missing-value imputation
- Categorical encoding
- Feature scaling
- Fixed train-test split
- Linear Regression
- Logistic Regression
- Training-time measurement
- Prediction-time measurement

Regression metrics:

- MAE
- RMSE
- R2

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score

## Part B - From Scratch

The manual implementation uses NumPy and Pandas without Scikit-learn
machine-learning utilities.

Implemented manually:

- Missing-value handling
- Categorical encoding
- Feature scaling
- Linear Regression
- Logistic Regression
- Sigmoid function
- Probability prediction
- Classification thresholding
- Gradient-descent optimization
- MAE
- RMSE
- R2
- Accuracy
- Precision
- Recall
- F1-score

The same train-test samples are reused for comparison.

## Part C - Comparison and Optimization

The manual Logistic Regression implementation was optimized by testing
different learning rates and numbers of epochs.

The optimization experiment is stored in:

results/logistic_optimization_results.csv

The final comparison is stored in:

results/final_comparison.csv

## Project Structure

```text
202618019_LAB05/
│
├── data/
│   └── garments_worker_productivity.csv
│
├── results/
│   ├── final_comparison.csv
│   ├── full_comparison.csv
│   ├── logistic_optimization_results.csv
│   ├── optimized_logistic_regression.csv
│   ├── sklearn_classification_results.csv
│   └── sklearn_regression_results.csv
│
├── lab05.ipynb
├── lab5_sklearn.py
├── lab5_from_scratch.py
├── common.py
├── run_lab5.py
├── requirements.txt
└── README.md