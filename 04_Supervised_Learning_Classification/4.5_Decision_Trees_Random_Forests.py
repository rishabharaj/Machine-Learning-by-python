"""
Decision Trees and Random Forests

This script demonstrates:
- Implementing Decision Trees
- Understanding tree structure and visualization
- Implementing Random Forests
- Feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load and prepare Iris dataset"""
    print("\n=== Loading Dataset ===")
    
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names

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

def train_decision_tree(X_train, y_train, feature_names, target_names):
    """Train and visualize decision tree"""
    print("\n=== Training Decision Tree ===")
    
    # Initialize and train model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Visualize tree
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True)
    
    plt.tight_layout()
    plt.savefig('decision_tree.png')
    plt.close()
    
    return model

def train_random_forest(X_train, y_train):
    """Train random forest classifier"""
    print("\n=== Training Random Forest ===")
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def tune_hyperparameters(X_train, y_train):
    """Tune random forest hyperparameters"""
    print("\n=== Tuning Hyperparameters ===")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize grid search
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
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

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n=== Evaluating {model_name} ===")
    
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

def plot_feature_importance(model, feature_names, title):
    """Plot feature importance"""
    print(f"\n=== Plotting Feature Importance for {title} ===")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importance - {title}')
    
    plt.tight_layout()
    plt.savefig(f'feature_importance_{title.lower().replace(" ", "_")}.png')
    plt.close()
    
    return importance_df

def compare_models(X_train, X_test, y_train, y_test):
    """Compare decision tree and random forest"""
    print("\n=== Comparing Models ===")
    
    # Train models
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit models
    tree.fit(X_train, y_train)
    forest.fit(X_train, y_train)
    
    # Calculate scores
    tree_score = tree.score(X_test, y_test)
    forest_score = forest.score(X_test, y_test)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Decision Tree', 'Random Forest'],
                y=[tree_score, forest_score])
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print(f"Decision Tree Accuracy: {tree_score:.4f}")
    print(f"Random Forest Accuracy: {forest_score:.4f}")

def main():
    """Main function to demonstrate decision trees and random forests"""
    print("=== Decision Trees and Random Forests ===")
    
    # Load and prepare data
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    
    # Train and visualize decision tree
    tree_model = train_decision_tree(X_train, y_train, feature_names, target_names)
    y_pred_tree = evaluate_model(tree_model, X_test, y_test, "Decision Tree")
    tree_importance = plot_feature_importance(tree_model, feature_names, "Decision Tree")
    
    # Train random forest
    forest_model = train_random_forest(X_train, y_train)
    y_pred_forest = evaluate_model(forest_model, X_test, y_test, "Random Forest")
    forest_importance = plot_feature_importance(forest_model, feature_names, "Random Forest")
    
    # Tune hyperparameters
    best_model = tune_hyperparameters(X_train, y_train)
    y_pred_best = evaluate_model(best_model, X_test, y_test, "Tuned Random Forest")
    best_importance = plot_feature_importance(best_model, feature_names, "Tuned Random Forest")
    
    # Compare models
    compare_models(X_train, X_test, y_train, y_test)
    
    print("\nAll examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 