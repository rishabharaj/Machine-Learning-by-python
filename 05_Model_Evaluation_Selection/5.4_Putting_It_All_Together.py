"""
Complete ML Pipeline

This script demonstrates a complete machine learning pipeline:
- Data loading and preprocessing
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Model deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_and_preprocess_data():
    """Load and preprocess the Boston Housing dataset"""
    print("\n=== Loading and Preprocessing Data ===")
    
    # Load dataset
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def create_feature_engineering_pipeline():
    """Create a pipeline for feature engineering"""
    print("\n=== Creating Feature Engineering Pipeline ===")
    
    # Define preprocessing steps
    preprocessing = Pipeline([
        ('scaler', StandardScaler()),
        ('polynomial', PolynomialFeatures(degree=2, include_bias=False))
    ])
    
    return preprocessing

def train_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessing):
    """Train and evaluate the model"""
    print("\n=== Training and Evaluating Model ===")
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 5, 10],
        'model__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print("Best parameters:", grid_search.best_params_)
    print(f"Training MSE: {train_mse:.3f}")
    print(f"Test MSE: {test_mse:.3f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    return best_model, y_train_pred, y_test_pred

def plot_results(y_train, y_train_pred, y_test, y_test_pred):
    """Plot actual vs predicted values"""
    print("\n=== Plotting Results ===")
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot training results
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title('Training Set: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Plot test results
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Test Set: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.close()

def save_model(model):
    """Save the trained model"""
    print("\n=== Saving Model ===")
    
    # Save model
    joblib.dump(model, 'boston_housing_model.joblib')
    print("Model saved as 'boston_housing_model.joblib'")

def main():
    """Main function to demonstrate the complete ML pipeline"""
    print("=== Complete ML Pipeline ===")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Create feature engineering pipeline
    preprocessing = create_feature_engineering_pipeline()
    
    # Train and evaluate model
    model, y_train_pred, y_test_pred = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessing
    )
    
    # Plot results
    plot_results(y_train, y_train_pred, y_test, y_test_pred)
    
    # Save model
    save_model(model)
    
    print("\nAll pipeline steps completed successfully!")
    print("Results have been saved as PNG files and the model has been saved.")

if __name__ == "__main__":
    main() 