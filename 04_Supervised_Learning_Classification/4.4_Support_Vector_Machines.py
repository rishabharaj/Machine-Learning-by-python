"""
Support Vector Machines (SVM)

This script demonstrates:
- Implementing SVM for classification
- Understanding kernel functions
- Hyperparameter tuning
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def generate_data():
    """Generate sample data for SVM demonstration"""
    print("\n=== Generating Data ===")
    
    # Generate linearly separable data
    X_linear, y_linear = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )
    
    # Generate non-linearly separable data (moons)
    X_moons, y_moons = make_moons(
        n_samples=200,
        noise=0.2,
        random_state=42
    )
    
    return X_linear, y_linear, X_moons, y_moons

def prepare_data(X, y):
    """Prepare data for SVM"""
    print("\n=== Preparing Data ===")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_linear_svm(X_train, y_train):
    """Train linear SVM"""
    print("\n=== Training Linear SVM ===")
    
    # Initialize and train model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    
    # Print support vectors
    print(f"Number of support vectors: {len(model.support_vectors_)}")
    
    return model

def train_rbf_svm(X_train, y_train):
    """Train RBF kernel SVM"""
    print("\n=== Training RBF Kernel SVM ===")
    
    # Initialize and train model
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)
    
    # Print support vectors
    print(f"Number of support vectors: {len(model.support_vectors_)}")
    
    return model

def tune_hyperparameters(X_train, y_train):
    """Tune SVM hyperparameters"""
    print("\n=== Tuning Hyperparameters ===")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Initialize grid search
    grid = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    # Perform grid search
    grid.fit(X_train, y_train)
    
    # Print best parameters
    print("Best parameters found:")
    print(grid.best_params_)
    
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n=== Evaluating Model ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    return y_pred

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary"""
    print(f"\n=== Plotting Decision Boundary for {title} ===")
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Predict for mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
    
    # Plot support vectors
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none',
                   edgecolors='k', label='Support Vectors')
    
    plt.title(f'Decision Boundary - {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.lower().replace(" ", "_")}.png')
    plt.close()

def compare_kernels(X_train, y_train, X_test, y_test):
    """Compare different kernel functions"""
    print("\n=== Comparing Kernel Functions ===")
    
    # Define kernels
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    scores = []
    
    # Calculate scores for each kernel
    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x=kernels, y=scores)
    plt.title('Comparison of Kernel Functions')
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png')
    plt.close()
    
    # Print results
    for kernel, score in zip(kernels, scores):
        print(f"{kernel}: {score:.4f}")

def main():
    """Main function to demonstrate SVM"""
    print("=== Support Vector Machines ===")
    
    # Generate data
    X_linear, y_linear, X_moons, y_moons = generate_data()
    
    # Process linear data
    X_train_linear, X_test_linear, y_train_linear, y_test_linear, _ = prepare_data(X_linear, y_linear)
    linear_model = train_linear_svm(X_train_linear, y_train_linear)
    y_pred_linear = evaluate_model(linear_model, X_test_linear, y_test_linear)
    plot_decision_boundary(linear_model, X_linear, y_linear, "Linear SVM")
    
    # Process non-linear data
    X_train_moons, X_test_moons, y_train_moons, y_test_moons, _ = prepare_data(X_moons, y_moons)
    rbf_model = train_rbf_svm(X_train_moons, y_train_moons)
    y_pred_moons = evaluate_model(rbf_model, X_test_moons, y_test_moons)
    plot_decision_boundary(rbf_model, X_moons, y_moons, "RBF Kernel SVM")
    
    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train_moons, y_train_moons)
    y_pred_best = evaluate_model(best_model, X_test_moons, y_test_moons)
    plot_decision_boundary(best_model, X_moons, y_moons, "Tuned SVM")
    
    # Compare kernels
    compare_kernels(X_train_moons, y_train_moons, X_test_moons, y_test_moons)
    
    print("\nAll SVM examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 