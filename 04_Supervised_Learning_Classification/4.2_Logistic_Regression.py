"""
Logistic Regression

This script demonstrates:
- Implementing Logistic Regression for binary classification
- Understanding the sigmoid function
- Model evaluation and interpretation
- Decision boundary visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_data():
    """Generate sample data for logistic regression"""
    print("\n=== Generating Data ===")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )
    
    return X, y

def plot_sigmoid():
    """Plot the sigmoid function"""
    print("\n=== Plotting Sigmoid Function ===")
    
    # Generate x values
    x = np.linspace(-10, 10, 100)
    
    # Calculate sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, sigmoid)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('σ(x)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sigmoid_function.png')
    plt.close()

def prepare_data(X, y):
    """Prepare data for logistic regression"""
    print("\n=== Preparing Data ===")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train logistic regression model"""
    print("\n=== Training Model ===")
    
    # Initialize and train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print(f"Intercept: {model.intercept_[0]:.4f}")
    print("\nCoefficients:")
    for i, coef in enumerate(model.coef_[0]):
        print(f"Feature {i}: {coef:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n=== Evaluating Model ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    return y_pred, y_prob

def plot_roc_curve(y_test, y_prob):
    """Plot ROC curve"""
    print("\n=== Plotting ROC Curve ===")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_test, y_prob):
    """Plot precision-recall curve"""
    print("\n=== Plotting Precision-Recall Curve ===")
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_decision_boundary(model, X, y):
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
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('decision_boundary.png')
    plt.close()

def main():
    """Main function to demonstrate logistic regression"""
    print("=== Logistic Regression ===")
    
    # Generate and prepare data
    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Plot sigmoid function
    plot_sigmoid()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred, y_prob = evaluate_model(model, X_test, y_test)
    
    # Plot curves
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    
    # Plot decision boundary
    plot_decision_boundary(model, X, y)
    
    print("\nAll logistic regression examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 