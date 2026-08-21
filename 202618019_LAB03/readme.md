
# DS605: Fundamentals of Machine Learning

## Lab Assignment 3

### Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Jayesh Hedaoo
**Student ID:** 202618019

---

## Objective

The objective of this lab is to build and compare Scikit-learn preprocessing pipelines and evaluate two classification models for predicting whether a hotel booking is canceled.

The target variable used for classification is `is_canceled`.

---

## Dataset

**Dataset:** Hotel Booking Demand

The dataset contains hotel booking information for City Hotel and Resort Hotel, including booking characteristics, stay information, customer details, and cancellation information.

**Dataset Source:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

**Original dataset:** Hotel Booking Demand Datasets by Nuno Antonio, Ana Almeida, and Luis Nunes.

The dataset used in this assignment contains **119,390 rows and 32 columns** before preprocessing.

---

## Part A - Data Loading and Preprocessing

### 1. Data Understanding

The following operations were performed:

* Loaded the Hotel Booking Demand dataset.
* Examined the first few records using `head()`.
* Checked dataset dimensions using `shape`.
* Examined data types using `dtypes`.
* Used `info()` to inspect the dataset structure.
* Used `describe()` for statistical analysis.
* Examined the class distribution of `is_canceled`.

### Target Variable Distribution

| `is_canceled` |  Count |
| ------------- | -----: |
| 0             | 75,166 |
| 1             | 44,224 |

The target variable `is_canceled` was used as the classification target.

---

## 2. Missing Values, Leakage and Outliers

### Missing Values

Missing-value counts and percentages were calculated for every column.

The `company` column contained a very high proportion of missing values and was removed from the modeling dataset.

Other columns containing missing values were retained and handled using the preprocessing pipelines.

### Data Leakage

The following columns were removed because they directly reveal the final booking outcome:

* `reservation_status`
* `reservation_status_date`

The `company` column was also removed because of its high level of missingness.

### Outlier Analysis

Outliers were investigated using boxplots and the IQR method.

The `adr` feature was inspected for extreme values. A negative ADR value was identified as invalid, and clearly extreme ADR values were removed.

Normal statistical outliers in count-based variables were not automatically removed because many of these values can represent legitimate hotel bookings.

The cleaned dataset was saved as:

```text
hotel_bookings_cleaned.csv
```

---

# Part A - Preprocessing Pipelines

The dataset was divided into numerical and categorical features.

### Numerical Preprocessing

KNN imputation with:

```text
KNNImputer(n_neighbors=5)
```

was used to handle missing numerical values.

### Categorical Preprocessing

Categorical missing values were handled using:

```text
SimpleImputer(strategy="most_frequent")
```

followed by:

```text
OneHotEncoder(handle_unknown="ignore")
```

---

## Pipeline A

For numerical features:

```text
KNNImputer → StandardScaler
```

For categorical features:

```text
Most Frequent Imputation → OneHotEncoder
```

---

## Pipeline B

For numerical features:

```text
KNNImputer → MinMaxScaler
```

For categorical features:

```text
Most Frequent Imputation → OneHotEncoder
```

A `ColumnTransformer` and Scikit-learn `Pipeline` were used to ensure that preprocessing was fitted only on the training data.

The dataset was split using:

```text
test_size = 0.2
stratify = y
random_state = 42
```

---

# Part B - Model Training and Evaluation

Two classification algorithms were trained using both preprocessing pipelines.

### Models

1. Logistic Regression
2. Decision Tree Classifier

Logistic Regression was trained using:

```text
LogisticRegression(max_iter=1000)
```

Decision Tree was trained using:

```text
DecisionTreeClassifier(random_state=42)
```

This resulted in four model-pipeline combinations.

---

# Model Comparison

| Model                            | Training Accuracy | Testing Accuracy | Precision | Recall |   F1-Score |
| -------------------------------- | ----------------: | ---------------: | --------: | -----: | ---------: |
| Logistic Regression + Pipeline A |            0.8190 |           0.8166 |    0.8081 | 0.6620 |     0.7278 |
| Logistic Regression + Pipeline B |            0.8154 |           0.8115 |    0.8021 | 0.6520 |     0.7193 |
| Decision Tree + Pipeline A       |            0.9962 |       **0.8563** |    0.8022 | 0.8123 | **0.8072** |
| Decision Tree + Pipeline B       |            0.9962 |           0.8561 |    0.8017 | 0.8124 |     0.8070 |

---

# Confusion Matrices

Confusion matrices were generated for:

* Best Logistic Regression: **Logistic Regression + Pipeline A**
* Best Decision Tree: **Decision Tree + Pipeline A**

The figures are available in the `figures/` directory.

```text
figures/
├── confusion_matrix_logistic_regression.png
└── confusion_matrix_decision_tree.png
```

---

# Final Observations

### 1. Best Overall Combination

Decision Tree with Pipeline A gives the best overall result. It achieved the highest testing accuracy of **85.63%** and the highest F1-score of **0.8072** among the four experiments.

### 2. StandardScaler vs MinMaxScaler for Logistic Regression

StandardScaler performed better than MinMaxScaler for Logistic Regression.

Pipeline A achieved:

* Testing Accuracy: **0.8166**
* F1-Score: **0.7278**

Pipeline B achieved:

* Testing Accuracy: **0.8115**
* F1-Score: **0.7193**

Therefore, scaling choice had a noticeable but relatively small effect on Logistic Regression.

### 3. Scaling and Decision Tree

Scaling made very little difference to the Decision Tree.

The testing accuracy was:

* Pipeline A: **0.8563**
* Pipeline B: **0.8561**

This shows that StandardScaler and MinMaxScaler produced almost identical Decision Tree performance.

### 4. Overfitting

The Decision Tree achieved a training accuracy of **99.62%**, while its testing accuracy was approximately **85.6%**.

The large difference between training and testing accuracy indicates possible overfitting.

### 5. Overall Model Comparison

Logistic Regression showed a smaller difference between training and testing performance, indicating more stable generalization.

However, the Decision Tree achieved better testing accuracy, recall, and F1-score. Therefore, **Decision Tree + Pipeline A** was the best-performing combination in this experiment.

---

# Repository Contents

```text
202618019_LAB03/
│
├── LAB3.ipynb
├── README.md
├── hotel_dataset.csv
├── hotel_bookings_cleaned.csv
├── model_comparison.csv
│
└── figures/
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_decision_tree.png
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Conclusion

This lab demonstrated the use of Scikit-learn preprocessing pipelines for handling missing values, scaling numerical features, and encoding categorical features. Logistic Regression and Decision Tree models were trained and evaluated using the same train-test split.

Among the four experiments, **Decision Tree with Pipeline A (KNNImputer + StandardScaler)** achieved the best overall testing performance with a testing accuracy of **85.63%** and an F1-score of **0.8072**.
