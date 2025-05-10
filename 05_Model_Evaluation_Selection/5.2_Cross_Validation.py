"""
Cross-Validation Techniques

This script demonstrates:
- K-Fold Cross-Validation
- Stratified K-Fold Cross-Validation
- Leave-One-Out Cross-Validation
- Time Series Cross-Validation
- Cross-validation with different scoring metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    TimeSeriesSplit, cross_val_score,
    cross_validate, learning_curve
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    make_scorer, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def generate_data(problem_type='regression'):
    """Generate sample data for cross-validation"""
    print("\n=== Generating Data ===")
    
    if problem_type == 'regression':
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=10,
            random_state=42
        )
    else:
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_classes=2,
            weights=[0.7, 0.3],
            random_state=42
        )
    
    return X, y

def prepare_data(X, y):
    """Prepare data for cross-validation"""
    print("\n=== Preparing Data ===")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def k_fold_cross_validation(X, y, problem_type='regression'):
    """Demonstrate K-Fold Cross-Validation"""
    print("\n=== K-Fold Cross-Validation ===")
    
    # Initialize model
    if problem_type == 'regression':
        model = RandomForestRegressor(random_state=42)
        scoring = 'neg_mean_squared_error'
    else:
        model = RandomForestClassifier(random_state=42)
        scoring = 'accuracy'
    
    # Perform K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)
    
    # Print results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    
    return scores

def stratified_k_fold_cross_validation(X, y):
    """Demonstrate Stratified K-Fold Cross-Validation"""
    print("\n=== Stratified K-Fold Cross-Validation ===")
    
    # Initialize model
    model = RandomForestClassifier(random_state=42)
    
    # Perform Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    # Print results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    
    return scores

def leave_one_out_cross_validation(X, y, problem_type='regression'):
    """Demonstrate Leave-One-Out Cross-Validation"""
    print("\n=== Leave-One-Out Cross-Validation ===")
    
    # Initialize model
    if problem_type == 'regression':
        model = RandomForestRegressor(random_state=42)
        scoring = 'neg_mean_squared_error'
    else:
        model = RandomForestClassifier(random_state=42)
        scoring = 'accuracy'
    
    # Perform LOOCV
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)
    
    # Print results
    print(f"Number of folds: {len(scores)}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    
    return scores

def time_series_cross_validation(X, y):
    """Demonstrate Time Series Cross-Validation"""
    print("\n=== Time Series Cross-Validation ===")
    
    # Initialize model
    model = RandomForestRegressor(random_state=42)
    
    # Perform Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    
    # Print results
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    
    return scores

def multiple_metrics_cross_validation(X, y):
    """Demonstrate cross-validation with multiple metrics"""
    print("\n=== Multiple Metrics Cross-Validation ===")
    
    # Initialize model
    model = RandomForestClassifier(random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    
    # Print results
    for metric, scores in cv_results.items():
        if metric.startswith('test_'):
            print(f"\n{metric[5:]}:")
            print(f"Scores: {scores}")
            print(f"Mean: {scores.mean():.4f}")
            print(f"Standard deviation: {scores.std():.4f}")
    
    return cv_results

def plot_learning_curve(X, y, problem_type='regression'):
    """Plot learning curve"""
    print("\n=== Plotting Learning Curve ===")
    
    # Initialize model
    if problem_type == 'regression':
        model = RandomForestRegressor(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

def main():
    """Main function to demonstrate cross-validation techniques"""
    print("=== Cross-Validation Techniques ===")
    
    # Generate and prepare regression data
    X_reg, y_reg = generate_data('regression')
    X_reg_scaled, y_reg = prepare_data(X_reg, y_reg)
    
    # Generate and prepare classification data
    X_clf, y_clf = generate_data('classification')
    X_clf_scaled, y_clf = prepare_data(X_clf, y_clf)
    
    # Demonstrate different cross-validation techniques
    k_fold_cross_validation(X_reg_scaled, y_reg, 'regression')
    k_fold_cross_validation(X_clf_scaled, y_clf, 'classification')
    
    stratified_k_fold_cross_validation(X_clf_scaled, y_clf)
    
    leave_one_out_cross_validation(X_reg_scaled, y_reg, 'regression')
    leave_one_out_cross_validation(X_clf_scaled, y_clf, 'classification')
    
    time_series_cross_validation(X_reg_scaled, y_reg)
    
    multiple_metrics_cross_validation(X_clf_scaled, y_clf)
    
    # Plot learning curves
    plot_learning_curve(X_reg_scaled, y_reg, 'regression')
    plot_learning_curve(X_clf_scaled, y_clf, 'classification')
    
    print("\nAll cross-validation examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 