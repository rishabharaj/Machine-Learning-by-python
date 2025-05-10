"""
K-Nearest Neighbors (KNN)

This script demonstrates:
- Implementing KNN classifier
- Understanding the 'k' parameter
- Distance metrics
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

def generate_data():
    """Generate sample data for KNN demonstration"""
    print("\n=== Generating Data ===")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    return X, y

def prepare_data(X, y):
    """Prepare data for KNN"""
    print("\n=== Preparing Data ===")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def find_optimal_k(X_train, y_train):
    """Find optimal k value using cross-validation"""
    print("\n=== Finding Optimal k ===")
    
    # Range of k values to try
    k_range = range(1, 31)
    k_scores = []
    
    # Calculate cross-validation scores
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    # Find optimal k
    optimal_k = k_range[np.argmax(k_scores)]
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of k')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('k vs Accuracy')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('k_vs_accuracy.png')
    plt.close()
    
    print(f"Optimal k value: {optimal_k}")
    return optimal_k

def train_model(X_train, y_train, k):
    """Train KNN model"""
    print("\n=== Training Model ===")
    
    # Initialize and train model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    return model

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

def plot_decision_boundary(model, X, y, k):
    """Plot decision boundary"""
    print("\n=== Plotting Decision Boundary ===")
    
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
    plt.title(f'Decision Boundary (k = {k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('decision_boundary.png')
    plt.close()

def compare_distance_metrics(X_train, y_train, X_test, y_test):
    """Compare different distance metrics"""
    print("\n=== Comparing Distance Metrics ===")
    
    # Define metrics
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    scores = []
    
    # Calculate scores for each metric
    for metric in metrics:
        model = KNeighborsClassifier(n_neighbors=5, metric=metric)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x=metrics, y=scores)
    plt.title('Comparison of Distance Metrics')
    plt.xlabel('Distance Metric')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('distance_metrics_comparison.png')
    plt.close()
    
    # Print results
    for metric, score in zip(metrics, scores):
        print(f"{metric}: {score:.4f}")

def main():
    """Main function to demonstrate KNN"""
    print("=== K-Nearest Neighbors ===")
    
    # Generate and prepare data
    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Find optimal k
    optimal_k = find_optimal_k(X_train, y_train)
    
    # Train model
    model = train_model(X_train, y_train, optimal_k)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y, optimal_k)
    
    # Compare distance metrics
    compare_distance_metrics(X_train, y_train, X_test, y_test)
    
    print("\nAll KNN examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 