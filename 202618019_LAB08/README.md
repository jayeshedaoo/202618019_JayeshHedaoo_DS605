# Lab 8: Decision Trees and Support Vector Machines

## Overview

This project implements Decision Trees and Linear Support Vector Machines (SVMs) from scratch and compares them with Scikit-learn implementations. It also explores nonlinear SVM kernels and evaluates the models on the Banknote Authentication dataset.

## Dataset

- Dataset: Banknote Authentication
- Features: Variance, Skewness, Curtosis, Entropy
- Target: Banknote class
- Dataset size: 1,372 samples

## Implementations

### Part A: Decision Tree from Scratch
- Gini impurity
- Entropy
- Weighted impurity calculation
- Best feature and threshold selection
- Recursive tree construction
- Prediction and decision-boundary visualization
- Comparison with Scikit-learn Decision Trees

### Part B: Linear SVM from Scratch
- Feature standardization
- Hinge loss optimization
- Gradient/subgradient-based training
- Prediction and decision-boundary visualization
- Comparison with Scikit-learn Linear SVM

### Part C: Nonlinear SVM Kernels
Experiments with Linear, Polynomial, and RBF kernels on synthetic datasets.

### Part D: Full Dataset Evaluation
Five models are compared using:
- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Prediction time

## Results

The Scikit-learn RBF SVM achieved 100% accuracy on the current test split. The Scikit-learn Decision Tree achieved 99.27%, while the Scikit-learn Linear SVM achieved 97.45%.

The manual Decision Tree achieved 98.18%, and the manual Linear SVM achieved 87.27%.

These results correspond to one train/test split and may vary with different data splits or hyperparameters.

## Project Structure

```text
202618019_LAB08/
├── LAB08.ipynb
├── README.md
└── results/
    ├── final_model_comparison.csv
    ├── decision_tree_rules.txt
    └── decision_tree_visualization.png
```

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- ucimlrepo, if used to download the dataset

## How to Run

1. Install the required libraries.
2. Open `LAB08.ipynb` in Jupyter Notebook or VS Code.
3. Run the notebook cells in order.
4. Review the generated plots and model comparison results.

## Conclusion

The experiments demonstrate the difference between interpretable tree-based models and margin-based classifiers. They also show how nonlinear SVM kernels can model complex decision boundaries. The manual implementations provide insight into how impurity-based splitting and hinge-loss optimization work internally.