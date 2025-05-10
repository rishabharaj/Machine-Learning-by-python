"""
Classification Evaluation

This script demonstrates:
- Various classification evaluation metrics
- Confusion matrix analysis
- ROC and PR curves
- Cross-validation techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, learning_curve
)
from sklearn.preprocessing import StandardScaler

def generate_data():
    """Generate sample data for classification"""
    print("\n=== Generating Data ===")
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )
    
    return X, y

def prepare_data(X, y):
    """Prepare data for classification"""
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
    """Train classification model"""
    print("\n=== Training Model ===")
    
    # Initialize and train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various evaluation metrics"""
    print("\n=== Calculating Metrics ===")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    print("\n=== Plotting Confusion Matrix ===")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve"""
    print("\n=== Plotting ROC Curve ===")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
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

def plot_precision_recall_curve(y_true, y_prob):
    """Plot precision-recall curve"""
    print("\n=== Plotting Precision-Recall Curve ===")
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'AP = {ap_score:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()

def perform_cross_validation(X, y):
    """Perform cross-validation"""
    print("\n=== Performing Cross-Validation ===")
    
    # Initialize model
    model = LogisticRegression(random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Print results
    print("Cross-validation scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    
    return cv_scores

def plot_learning_curve(X, y):
    """Plot learning curve"""
    print("\n=== Plotting Learning Curve ===")
    
    # Initialize model
    model = LogisticRegression(random_state=42)
    
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
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curve')
    plt.legend(loc="best")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

def main():
    """Main function to demonstrate classification evaluation"""
    print("=== Classification Evaluation ===")
    
    # Generate and prepare data
    X, y = generate_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    
    # Plot evaluation metrics
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    
    # Perform cross-validation
    cv_scores = perform_cross_validation(X, y)
    
    # Plot learning curve
    plot_learning_curve(X, y)
    
    print("\nAll evaluation examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 