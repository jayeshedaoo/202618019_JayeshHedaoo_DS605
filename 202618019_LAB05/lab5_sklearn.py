import os
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/garments_worker_productivity.csv"
RESULTS_DIR = "results"


os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# 2. Create classification target
# ---------------------------------------------------------

df["MeetsTarget"] = (
    df["actual_productivity"] >= df["targeted_productivity"]
).astype(int)


# ---------------------------------------------------------
# 3. Define features
# ---------------------------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column not in ["actual_productivity", "MeetsTarget"]
]

categorical_features = df[feature_columns].select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = df[feature_columns].select_dtypes(
    exclude=["object"]
).columns.tolist()


# ---------------------------------------------------------
# 4. Fixed train-test split
# ---------------------------------------------------------

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42
)

np.savez(
    os.path.join(RESULTS_DIR, "split_indices.npz"),
    train_idx=train_idx,
    test_idx=test_idx
)


# ---------------------------------------------------------
# 5. Prepare targets
# ---------------------------------------------------------

X = df[feature_columns]

y_regression = df["actual_productivity"]

y_classification = df["MeetsTarget"]


X_train = X.loc[train_idx]
X_test = X.loc[test_idx]

y_reg_train = y_regression.loc[train_idx]
y_reg_test = y_regression.loc[test_idx]

y_cls_train = y_classification.loc[train_idx]
y_cls_test = y_classification.loc[test_idx]


# ---------------------------------------------------------
# 6. Scikit-learn preprocessing
# ---------------------------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# ---------------------------------------------------------
# 7. Linear Regression
# ---------------------------------------------------------

regression_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])


start_time = time.perf_counter()

regression_model.fit(
    X_train,
    y_reg_train
)

regression_train_time = time.perf_counter() - start_time


start_time = time.perf_counter()

y_reg_pred = regression_model.predict(X_test)

regression_prediction_time = time.perf_counter() - start_time


regression_mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

regression_rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

regression_r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


sklearn_regression_results = {
    "Model": "Scikit-learn Linear Regression",
    "Task": "Regression",
    "MAE": regression_mae,
    "RMSE": regression_rmse,
    "R2": regression_r2,
    "Accuracy": np.nan,
    "Precision": np.nan,
    "Recall": np.nan,
    "F1": np.nan,
    "Training Time": regression_train_time,
    "Prediction Time": regression_prediction_time
}


# ---------------------------------------------------------
# 8. Logistic Regression
# ---------------------------------------------------------

classification_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


start_time = time.perf_counter()

classification_model.fit(
    X_train,
    y_cls_train
)

classification_train_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

y_cls_pred = classification_model.predict(X_test)

classification_prediction_time = (
    time.perf_counter() - start_time
)


classification_accuracy = accuracy_score(
    y_cls_test,
    y_cls_pred
)

classification_precision = precision_score(
    y_cls_test,
    y_cls_pred,
    zero_division=0
)

classification_recall = recall_score(
    y_cls_test,
    y_cls_pred,
    zero_division=0
)

classification_f1 = f1_score(
    y_cls_test,
    y_cls_pred,
    zero_division=0
)


sklearn_classification_results = {
    "Model": "Scikit-learn Logistic Regression",
    "Task": "Classification",
    "MAE": np.nan,
    "RMSE": np.nan,
    "R2": np.nan,
    "Accuracy": classification_accuracy,
    "Precision": classification_precision,
    "Recall": classification_recall,
    "F1": classification_f1,
    "Training Time": classification_train_time,
    "Prediction Time": classification_prediction_time
}


# ---------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------

regression_table = pd.DataFrame(
    [sklearn_regression_results]
)

classification_table = pd.DataFrame(
    [sklearn_classification_results]
)

regression_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "sklearn_regression_results.csv"
    ),
    index=False
)

classification_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "sklearn_classification_results.csv"
    ),
    index=False
)


print("\nScikit-learn Regression Results")
print(regression_table.to_string(index=False))

print("\nScikit-learn Classification Results")
print(classification_table.to_string(index=False))

print("\nSaved:")
print("results/split_indices.npz")
print("results/sklearn_regression_results.csv")
print("results/sklearn_classification_results.csv")