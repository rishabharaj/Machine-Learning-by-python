"""
Regression Evaluation

This script demonstrates:
- Various regression evaluation metrics
- Cross-validation techniques
- Residual analysis
- Model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

def load_data():
    """Load and prepare Boston Housing dataset"""
    print("\n=== Loading Dataset ===")
    
    # Load dataset
    boston = load_boston()
    
    # Create DataFrame
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target
    
    return df

def prepare_data(df):
    """Prepare data for regression"""
    print("\n=== Preparing Data ===")
    
    # Split features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def train_models(X, y):
    """Train different regression models"""
    print("\n=== Training Models ===")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X, y)
        print(f"\n{name} trained successfully")
    
    return models

def calculate_metrics(models, X, y):
    """Calculate various evaluation metrics"""
    print("\n=== Calculating Evaluation Metrics ===")
    
    # Initialize metrics dictionary
    metrics = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics[name] = {
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R²': r2_score(y, y_pred),
            'Explained Variance': explained_variance_score(y, y_pred),
            'Max Error': max_error(y, y_pred)
        }
    
    # Print metrics
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name} Metrics:")
        for metric_name, value in model_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    return metrics

def perform_cross_validation(models, X, y):
    """Perform k-fold cross-validation"""
    print("\n=== Performing Cross-Validation ===")
    
    # Initialize k-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store cross-validation scores
    cv_scores = {}
    
    for name, model in models.items():
        # Calculate cross-validation scores
        scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        cv_scores[name] = {
            'Mean MSE': -scores.mean(),
            'Std MSE': scores.std()
        }
        
        print(f"\n{name} Cross-Validation:")
        print(f"Mean MSE: {-scores.mean():.4f}")
        print(f"Standard Deviation: {scores.std():.4f}")
    
    return cv_scores

def analyze_residuals(models, X, y):
    """Analyze model residuals"""
    print("\n=== Analyzing Residuals ===")
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items(), 1):
        # Calculate residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Plot residuals
        plt.subplot(1, 3, i)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{name} Residuals')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png')
    plt.close()

def plot_metrics_comparison(metrics):
    """Plot comparison of model metrics"""
    print("\n=== Plotting Metrics Comparison ===")
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics).T
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(['MSE', 'RMSE', 'MAE', 'R²'], 1):
        plt.subplot(2, 2, i)
        sns.barplot(x=metrics_df.index, y=metric, data=metrics_df)
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def main():
    """Main function to demonstrate regression evaluation"""
    print("=== Regression Evaluation ===")
    
    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)
    
    # Train models
    models = train_models(X, y)
    
    # Calculate metrics
    metrics = calculate_metrics(models, X, y)
    
    # Perform cross-validation
    cv_scores = perform_cross_validation(models, X, y)
    
    # Analyze residuals
    analyze_residuals(models, X, y)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics)
    
    print("\nAll evaluation examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 