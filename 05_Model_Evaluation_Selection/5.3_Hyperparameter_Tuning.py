"""
Hyperparameter Tuning

This script demonstrates:
- Grid Search Cross-Validation
- Randomized Search Cross-Validation
- Bayesian Optimization
- Hyperparameter importance analysis
- Learning curve analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (
    train_test_split, GridSearchCV,
    RandomizedSearchCV, learning_curve
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, mean_squared_error,
    make_scorer, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

def generate_data(problem_type='regression'):
    """Generate sample data for hyperparameter tuning"""
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
    """Prepare data for hyperparameter tuning"""
    print("\n=== Preparing Data ===")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def grid_search_cv(X_train, y_train, problem_type='regression'):
    """Demonstrate Grid Search Cross-Validation"""
    print("\n=== Grid Search Cross-Validation ===")
    
    if problem_type == 'regression':
        # Define model and parameter grid
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        scoring = 'neg_mean_squared_error'
    else:
        # Define model and parameter grid
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        scoring = 'accuracy'
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"\nBest score: {grid_search.best_score_:.4f}")
    
    return grid_search

def randomized_search_cv(X_train, y_train, problem_type='regression'):
    """Demonstrate Randomized Search Cross-Validation"""
    print("\n=== Randomized Search Cross-Validation ===")
    
    if problem_type == 'regression':
        # Define model and parameter distributions
        model = SVR()
        param_dist = {
            'C': uniform(0.1, 10),
            'gamma': uniform(0.01, 1),
            'kernel': ['linear', 'rbf', 'poly']
        }
        scoring = 'neg_mean_squared_error'
    else:
        # Define model and parameter distributions
        model = SVC(random_state=42)
        param_dist = {
            'C': uniform(0.1, 10),
            'gamma': uniform(0.01, 1),
            'kernel': ['linear', 'rbf', 'poly']
        }
        scoring = 'accuracy'
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    # Print results
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest score: {random_search.best_score_:.4f}")
    
    return random_search

def plot_hyperparameter_importance(grid_search, param_grid):
    """Plot hyperparameter importance"""
    print("\n=== Plotting Hyperparameter Importance ===")
    
    # Get results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Plot parameter importance
    plt.figure(figsize=(12, 6))
    for param in param_grid.keys():
        param_values = results[f'param_{param}']
        mean_scores = results['mean_test_score']
        
        plt.scatter(param_values, mean_scores, label=param)
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Mean Test Score')
    plt.title('Hyperparameter Importance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_importance.png')
    plt.close()

def plot_learning_curves(model, X_train, y_train):
    """Plot learning curves"""
    print("\n=== Plotting Learning Curves ===")
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5,
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
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
    """Main function to demonstrate hyperparameter tuning"""
    print("=== Hyperparameter Tuning ===")
    
    # Generate and prepare regression data
    X_reg, y_reg = generate_data('regression')
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_data(X_reg, y_reg)
    
    # Generate and prepare classification data
    X_clf, y_clf = generate_data('classification')
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = prepare_data(X_clf, y_clf)
    
    # Demonstrate grid search
    grid_search_reg = grid_search_cv(X_train_reg, y_train_reg, 'regression')
    grid_search_clf = grid_search_cv(X_train_clf, y_train_clf, 'classification')
    
    # Demonstrate randomized search
    random_search_reg = randomized_search_cv(X_train_reg, y_train_reg, 'regression')
    random_search_clf = randomized_search_cv(X_train_clf, y_train_clf, 'classification')
    
    # Plot hyperparameter importance
    param_grid_reg = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    plot_hyperparameter_importance(grid_search_reg, param_grid_reg)
    
    # Plot learning curves
    plot_learning_curves(grid_search_reg.best_estimator_, X_train_reg, y_train_reg)
    plot_learning_curves(grid_search_clf.best_estimator_, X_train_clf, y_train_clf)
    
    print("\nAll hyperparameter tuning examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 