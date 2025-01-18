# MNIST Digit Classification

This project implements handwritten digit classification on the MNIST dataset using Logistic Regression and Support Vector Machines (SVM). The repository includes the following:

## Features
- **Preprocessing**: Data normalization, feature selection, and dataset splitting (training, validation, testing).
- **Logistic Regression**: One-vs-all implementation with custom gradient descent optimization.
- **Support Vector Machines (SVM)**:
  - Linear kernel.
  - Radial Basis Function (RBF) kernel with hyperparameter tuning (`gamma` and `C`).
- **Performance Analysis**:
  - Accuracy evaluation for training, validation, and testing sets.
  - Graphical analysis of hyperparameter impact on performance.

## Results Summary
- Logistic Regression achieved ~91.86% accuracy on the test set.
- SVM with an RBF kernel (default gamma) achieved the highest accuracy of ~97.87%.
- Comparative analysis highlights the strengths and weaknesses of each method.

## Usage
- The project includes a Python script for running the models and evaluating results.
- All required data preprocessing steps and functions are included.

## Additional Details
- Code is written in Python, utilizing `scipy`, `numpy`, `matplotlib`, and `sklearn`.
- The repository also includes a detailed report summarizing methodology, results, and insights.
