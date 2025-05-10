"""
Introduction to Classification

This script demonstrates:
- Basic concepts of classification
- Visualizing decision boundaries
- Understanding classification metrics
- Simple classification using scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def generate_sample_data():
    """Generate sample data for classification demonstration"""
    print("\n=== Generating Sample Data ===")
    
    # Generate linearly separable data
    X_linear, y_linear = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Generate non-linearly separable data (moons)
    X_moons, y_moons = make_moons(
        n_samples=200,
        noise=0.2,
        random_state=42
    )
    
    return X_linear, y_linear, X_moons, y_moons

def visualize_data(X_linear, y_linear, X_moons, y_moons):
    """Visualize the classification datasets"""
    print("\n=== Visualizing Data ===")
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot linearly separable data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_linear[:, 0], y=X_linear[:, 1], hue=y_linear)
    plt.title('Linearly Separable Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot non-linearly separable data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_moons[:, 0], y=X_moons[:, 1], hue=y_moons)
    plt.title('Non-linearly Separable Data (Moons)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('classification_data.png')
    plt.close()

def train_classifier(X, y):
    """Train a simple classifier"""
    print("\n=== Training Classifier ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    return model, X_train, X_test, y_train, y_test

def plot_decision_boundary(model, X, y, title):
    """Plot decision boundary of classifier"""
    print(f"\n=== Plotting Decision Boundary for {title} ===")
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Predict for mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
    plt.title(f'Decision Boundary - {title}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.lower().replace(" ", "_")}.png')
    plt.close()

def demonstrate_classification_metrics(y_true, y_pred):
    """Demonstrate classification metrics"""
    print("\n=== Classification Metrics ===")
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def main():
    """Main function to demonstrate classification concepts"""
    print("=== Introduction to Classification ===")
    
    # Generate and visualize data
    X_linear, y_linear, X_moons, y_moons = generate_sample_data()
    visualize_data(X_linear, y_linear, X_moons, y_moons)
    
    # Train and evaluate on linear data
    model_linear, X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_classifier(X_linear, y_linear)
    plot_decision_boundary(model_linear, X_linear, y_linear, "Linear Data")
    
    # Train and evaluate on non-linear data
    model_moons, X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_classifier(X_moons, y_moons)
    plot_decision_boundary(model_moons, X_moons, y_moons, "Non-linear Data")
    
    # Demonstrate metrics
    y_pred_linear = model_linear.predict(X_test_linear)
    demonstrate_classification_metrics(y_test_linear, y_pred_linear)
    
    print("\nAll classification examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 