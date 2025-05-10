"""
Simple Linear Regression

This script demonstrates:
- Implementing simple linear regression with one feature
- Visualizing the regression line
- Making predictions
- Evaluating model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

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

def prepare_data(df):
    """Prepare data for simple linear regression"""
    print("\n=== Preparing Data ===")
    
    # Select feature with highest correlation to target
    correlation = df.corr()['MEDV'].abs()
    feature = correlation.drop('MEDV').idxmax()
    
    print(f"Selected feature: {feature}")
    print(f"Correlation with MEDV: {correlation[feature]:.4f}")
    
    # Prepare X and y
    X = df[[feature]]
    y = df['MEDV']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, feature

def train_model(X_train, y_train):
    """Train simple linear regression model"""
    print("\n=== Training Model ===")
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print(f"Slope (coefficient): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
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

def visualize_results(model, X_train, X_test, y_train, y_test, feature):
    """Visualize regression results"""
    print("\n=== Visualizing Results ===")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Training data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_train[feature], y=y_train, label='Actual')
    plt.plot(X_train[feature], model.predict(X_train), color='red', label='Predicted')
    plt.title('Training Data')
    plt.xlabel(feature)
    plt.ylabel('MEDV')
    plt.legend()
    
    # Test data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_test[feature], y=y_test, label='Actual')
    plt.plot(X_test[feature], model.predict(X_test), color='red', label='Predicted')
    plt.title('Test Data')
    plt.xlabel(feature)
    plt.ylabel('MEDV')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('regression_results.png')
    plt.close()

def make_predictions(model, X_test, y_test):
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
    
    return results

def main():
    """Main function to demonstrate simple linear regression"""
    print("=== Simple Linear Regression ===")
    
    # Load and prepare data
    df = load_boston_data()
    explore_data(df)
    X_train, X_test, y_train, y_test, feature = prepare_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(model, X_train, X_test, y_train, y_test, feature)
    
    # Make predictions
    results = make_predictions(model, X_test, y_test)
    
    print("\nAll regression examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 