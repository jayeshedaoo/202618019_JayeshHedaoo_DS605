import os
import time

import numpy as np
import pandas as pd

from common import (
    manual_accuracy,
    manual_f1,
    manual_mae,
    manual_precision,
    manual_r2,
    manual_recall,
    manual_rmse,
    sigmoid,
    train_logistic_regression,
)


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
# 3. Load EXACT SAME train-test split
# ---------------------------------------------------------

split_data = np.load(
    os.path.join(
        RESULTS_DIR,
        "split_indices.npz"
    )
)

train_idx = split_data["train_idx"]
test_idx = split_data["test_idx"]


# ---------------------------------------------------------
# 4. Define features
# ---------------------------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column not in ["actual_productivity", "MeetsTarget"]
]

categorical_features = df[feature_columns].select_dtypes(
    include=["object"]
).columns.tolist()


# ---------------------------------------------------------
# 5. Manual preprocessing
# ---------------------------------------------------------

manual_X = df[feature_columns].copy()


# One-hot encode categorical columns
manual_X = pd.get_dummies(
    manual_X,
    columns=categorical_features,
    drop_first=False
)


manual_X = manual_X.astype(float)


# Split BEFORE calculating preprocessing statistics
X_train = manual_X.loc[train_idx].copy()
X_test = manual_X.loc[test_idx].copy()


# ---------------------------------------------------------
# 6. Manual missing-value handling
# ---------------------------------------------------------

for column in X_train.columns:

    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(
        median_value
    )

    X_test[column] = X_test[column].fillna(
        median_value
    )


# ---------------------------------------------------------
# 7. Manual feature scaling
# ---------------------------------------------------------

train_mean = X_train.mean()

# ddof=0 matches StandardScaler's population standard deviation
train_std = X_train.std(ddof=0)

train_std = train_std.replace(0, 1)


X_train = (
    X_train - train_mean
) / train_std


X_test = (
    X_test - train_mean
) / train_std


X_train = X_train.to_numpy()

X_test = X_test.to_numpy()


# ---------------------------------------------------------
# 8. Prepare targets
# ---------------------------------------------------------

y_regression = df[
    "actual_productivity"
].to_numpy()

y_classification = df[
    "MeetsTarget"
].to_numpy()


y_reg_train = y_regression[train_idx]

y_reg_test = y_regression[test_idx]

y_cls_train = y_classification[train_idx]

y_cls_test = y_classification[test_idx]


# ---------------------------------------------------------
# 9. Manual Linear Regression
# ---------------------------------------------------------

X_train_lr = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_lr = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]


start_time = time.perf_counter()

linear_weights = np.linalg.lstsq(
    X_train_lr,
    y_reg_train,
    rcond=None
)[0]

linear_training_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

y_reg_pred = (
    X_test_lr @ linear_weights
)

linear_prediction_time = (
    time.perf_counter() - start_time
)


linear_mae = manual_mae(
    y_reg_test,
    y_reg_pred
)

linear_rmse = manual_rmse(
    y_reg_test,
    y_reg_pred
)

linear_r2 = manual_r2(
    y_reg_test,
    y_reg_pred
)


manual_regression_results = {
    "Model": "From-Scratch Linear Regression",
    "Task": "Regression",
    "MAE": linear_mae,
    "RMSE": linear_rmse,
    "R2": linear_r2,
    "Accuracy": np.nan,
    "Precision": np.nan,
    "Recall": np.nan,
    "F1": np.nan,
    "Training Time": linear_training_time,
    "Prediction Time": linear_prediction_time
}


# ---------------------------------------------------------
# 10. Manual Logistic Regression
# ---------------------------------------------------------

X_train_log = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_log = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]


# Baseline configuration
learning_rate = 0.01
epochs = 2000


start_time = time.perf_counter()

logistic_weights = train_logistic_regression(
    X_train_log,
    y_cls_train,
    learning_rate=learning_rate,
    epochs=epochs
)

logistic_training_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

probabilities = sigmoid(
    X_test_log @ logistic_weights
)

predictions = (
    probabilities >= 0.5
).astype(int)

logistic_prediction_time = (
    time.perf_counter() - start_time
)


logistic_accuracy = manual_accuracy(
    y_cls_test,
    predictions
)

logistic_precision = manual_precision(
    y_cls_test,
    predictions
)

logistic_recall = manual_recall(
    y_cls_test,
    predictions
)

logistic_f1 = manual_f1(
    y_cls_test,
    predictions
)


manual_classification_results = {
    "Model": "From-Scratch Logistic Regression",
    "Task": "Classification",
    "MAE": np.nan,
    "RMSE": np.nan,
    "R2": np.nan,
    "Accuracy": logistic_accuracy,
    "Precision": logistic_precision,
    "Recall": logistic_recall,
    "F1": logistic_f1,
    "Training Time": logistic_training_time,
    "Prediction Time": logistic_prediction_time
}


# ---------------------------------------------------------
# 11. Optimization experiment
# ---------------------------------------------------------

learning_rates = [
    0.001,
    0.01,
    0.05,
    0.1
]

epochs_list = [
    1000,
    2000,
    5000
]


optimization_results = []


for lr in learning_rates:

    for number_of_epochs in epochs_list:

        start_time = time.perf_counter()

        weights = train_logistic_regression(
            X_train_log,
            y_cls_train,
            learning_rate=lr,
            epochs=number_of_epochs
        )

        training_time = (
            time.perf_counter() - start_time
        )

        probabilities = sigmoid(
            X_test_log @ weights
        )

        optimized_predictions = (
            probabilities >= 0.5
        ).astype(int)

        optimization_results.append({
            "Learning Rate": lr,
            "Epochs": number_of_epochs,
            "Accuracy": manual_accuracy(
                y_cls_test,
                optimized_predictions
            ),
            "Precision": manual_precision(
                y_cls_test,
                optimized_predictions
            ),
            "Recall": manual_recall(
                y_cls_test,
                optimized_predictions
            ),
            "F1": manual_f1(
                y_cls_test,
                optimized_predictions
            ),
            "Training Time": training_time
        })


optimization_df = pd.DataFrame(
    optimization_results
)


# ---------------------------------------------------------
# 12. Final optimized model
# ---------------------------------------------------------

optimized_learning_rate = 0.05
optimized_epochs = 1000


start_time = time.perf_counter()

optimized_weights = train_logistic_regression(
    X_train_log,
    y_cls_train,
    learning_rate=optimized_learning_rate,
    epochs=optimized_epochs
)

optimized_training_time = (
    time.perf_counter() - start_time
)


start_time = time.perf_counter()

optimized_probabilities = sigmoid(
    X_test_log @ optimized_weights
)

optimized_predictions = (
    optimized_probabilities >= 0.5
).astype(int)

optimized_prediction_time = (
    time.perf_counter() - start_time
)


optimized_accuracy = manual_accuracy(
    y_cls_test,
    optimized_predictions
)

optimized_precision = manual_precision(
    y_cls_test,
    optimized_predictions
)

optimized_recall = manual_recall(
    y_cls_test,
    optimized_predictions
)

optimized_f1 = manual_f1(
    y_cls_test,
    optimized_predictions
)


optimized_classification_results = {
    "Model": "Optimized From-Scratch Logistic Regression",
    "Task": "Classification",
    "MAE": np.nan,
    "RMSE": np.nan,
    "R2": np.nan,
    "Accuracy": optimized_accuracy,
    "Precision": optimized_precision,
    "Recall": optimized_recall,
    "F1": optimized_f1,
    "Training Time": optimized_training_time,
    "Prediction Time": optimized_prediction_time,
    "Learning Rate": optimized_learning_rate,
    "Epochs": optimized_epochs
}


# ---------------------------------------------------------
# 13. Save results
# ---------------------------------------------------------

manual_regression_table = pd.DataFrame(
    [manual_regression_results]
)

manual_classification_table = pd.DataFrame(
    [manual_classification_results]
)

optimized_table = pd.DataFrame(
    [optimized_classification_results]
)


manual_regression_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "manual_regression_results.csv"
    ),
    index=False
)

manual_classification_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "manual_classification_results.csv"
    ),
    index=False
)

optimization_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "logistic_optimization_results.csv"
    ),
    index=False
)

optimized_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "optimized_logistic_regression.csv"
    ),
    index=False
)


print("\nFrom-Scratch Regression")
print(manual_regression_table.to_string(index=False))

print("\nFrom-Scratch Classification")
print(manual_classification_table.to_string(index=False))

print("\nOptimized Classification")
print(optimized_table.to_string(index=False))

print("\nOptimization results saved.")