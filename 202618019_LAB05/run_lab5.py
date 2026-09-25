import os
import subprocess
import sys


print("=" * 60)
print("DS605 LAB 5")
print("Machine Learning with Scikit-learn and From Scratch")
print("=" * 60)


print("\n[1/2] Running Scikit-learn implementation...")

subprocess.run(
    [sys.executable, "lab5_sklearn.py"],
    check=True
)


print("\n[2/2] Running From-Scratch implementation...")

subprocess.run(
    [sys.executable, "lab5_from_scratch.py"],
    check=True
)


print("\nCreating final comparison...")


import pandas as pd


results_dir = "results"


sklearn_reg = pd.read_csv(
    os.path.join(
        results_dir,
        "sklearn_regression_results.csv"
    )
)

sklearn_cls = pd.read_csv(
    os.path.join(
        results_dir,
        "sklearn_classification_results.csv"
    )
)

manual_reg = pd.read_csv(
    os.path.join(
        results_dir,
        "manual_regression_results.csv"
    )
)

manual_cls = pd.read_csv(
    os.path.join(
        results_dir,
        "manual_classification_results.csv"
    )
)

optimized_cls = pd.read_csv(
    os.path.join(
        results_dir,
        "optimized_logistic_regression.csv"
    )
)


final_comparison = pd.concat(
    [
        sklearn_reg,
        manual_reg,
        sklearn_cls,
        manual_cls,
        optimized_cls
    ],
    ignore_index=True
)


final_comparison.to_csv(
    os.path.join(
        results_dir,
        "final_comparison.csv"
    ),
    index=False
)


print("\nFinal Comparison")
print("=" * 60)
print(final_comparison.to_string(index=False))

print("\nFinal comparison saved to:")
print("results/final_comparison.csv")

print("\nLab 5 completed successfully.")