"""
Multiple Linear Regression

This script demonstrates:
- Implementing multiple linear regression with multiple features
- Feature selection and importance
- Handling multicollinearity
- Model evaluation and interpretation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_boston_data():
    """Load and prepare Boston Housing dataset"""
    print("\n=== Loading Boston Housing Dataset ===")
    
    # Load dataset
    from sklearn.datasets import load_boston
    boston = load_boston()
    
    # Create DataFrame
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target  # Median value of owner-occupied homes
    
    return df

def explore_data(df):
    """Explore the dataset"""
    print("\n=== Exploring Dataset ===")
    
    # Display basic information
    print("\nDataset Info:")
    print(df.info())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def check_multicollinearity(X):
    """Check for multicollinearity using VIF"""
    print("\n=== Checking Multicollinearity ===")
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)
    
    # Plot VIF values
    plt.figure(figsize=(10, 6))
    sns.barplot(x="VIF", y="Feature", data=vif_data)
    plt.title('VIF Values for Features')
    plt.axvline(x=5, color='r', linestyle='--', label='VIF = 5')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vif_plot.png')
    plt.close()
    
    return vif_data

def prepare_data(df):
    """Prepare data for multiple linear regression"""
    print("\n=== Preparing Data ===")
    
    # Select features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train multiple linear regression model"""
    print("\n=== Training Model ===")
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print("\nModel Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"\nIntercept: {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n=== Evaluating Model ===")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred)
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_feature_importance(model, X_train):
    """Plot feature importance"""
    print("\n=== Plotting Feature Importance ===")
    
    # Get feature importance (absolute coefficients)
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(model.coef_)
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Feature Importance (Absolute Coefficients)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

def make_predictions(model, X_test, y_test, scaler):
    """Make and display predictions"""
    print("\n=== Making Predictions ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create DataFrame with actual and predicted values
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Difference': y_test - y_pred
    })
    
    # Display first 10 predictions
    print("\nFirst 10 predictions:")
    print(results.head(10))
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Actual', y='Predicted', data=results)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual MEDV')
    plt.ylabel('Predicted MEDV')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    return results

def main():
    """Main function to demonstrate multiple linear regression"""
    print("=== Multiple Linear Regression ===")
    
    # Load and prepare data
    df = load_boston_data()
    explore_data(df)
    
    # Check multicollinearity
    X = df.drop('MEDV', axis=1)
    vif_data = check_multicollinearity(X)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Plot feature importance
    importance = plot_feature_importance(model, X_train)
    
    # Make predictions
    results = make_predictions(model, X_test, y_test, scaler)
    
    print("\nAll regression examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 