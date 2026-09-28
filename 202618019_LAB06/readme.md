# Observations and Discussion

## Part A – Image Classification

The image dataset was converted into numerical features using image brightness, contrast, pixel ratios, intensity values, and Canny edge information. These features were used to train traditional machine learning classifiers.

The models were evaluated using accuracy, precision, recall, F1-score, confusion matrix, training time, and prediction time. The results show how different traditional classifiers perform when using handcrafted image features.

## Part B – Email Classification

The email dataset contains 5,172 emails with 3,000 precomputed word-count features. The dataset contains 3,672 non-spam emails and 1,500 spam emails.

Three traditional machine learning models were trained using the existing word-count representation:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest

Logistic Regression achieved an accuracy of approximately 98.26% and an F1-score of approximately 97.04%. Naive Bayes achieved approximately 94.20% accuracy and 90.42% F1-score, while Random Forest achieved approximately 96.43% accuracy and 93.86% F1-score.

The models were also compared based on training and prediction time.

Note: The provided email dataset is already represented as word-count features. Therefore, TF-IDF vectorization was not applied in this implementation.

## Part C – Feature Reduction Improvement

Feature selection was applied using Chi-Square (`SelectKBest`) to reduce the number of email features from 3,000 to 1,500.

The reduced-feature Logistic Regression model achieved approximately 97.58% accuracy and 95.85% F1-score.

Feature reduction decreased the dimensionality and prediction time, but the classification performance decreased slightly compared with the original 3,000-feature model. This demonstrates the trade-off between computational efficiency and predictive performance.