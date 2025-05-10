"""
Introduction to Regression

This script demonstrates:
- Basic concepts of regression analysis
- Visualizing relationships between variables
- Understanding the concept of a best-fit line
- Simple regression using scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def generate_sample_data():
    """Generate sample data for regression demonstration"""
    print("\n=== Generating Sample Data ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate x values
    x = np.linspace(0, 10, 100)
    
    # Generate y values with some noise
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    
    # Create DataFrame
    df = pd.DataFrame({'x': x, 'y': y})
    
    return df

def visualize_relationship(df):
    """Visualize the relationship between variables"""
    print("\n=== Visualizing Relationship ===")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y')
    plt.title('Relationship between X and Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # Save plot
    plt.savefig('relationship_plot.png')
    plt.close()

def fit_regression_line(df):
    """Fit and visualize regression line"""
    print("\n=== Fitting Regression Line ===")
    
    # Prepare data
    X = df[['x']]
    y = df['y']
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Print model parameters
    print(f"Slope (coefficient): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    # Create plot with regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', label='Data points')
    plt.plot(X, y_pred, color='red', label='Regression line')
    plt.title('Regression Line Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('regression_line.png')
    plt.close()
    
    return model

def demonstrate_residuals(df, model):
    """Demonstrate and visualize residuals"""
    print("\n=== Analyzing Residuals ===")
    
    # Calculate predictions and residuals
    X = df[['x']]
    y = df['y']
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create residual plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Save plot
    plt.savefig('residual_plot.png')
    plt.close()
    
    # Print residual statistics
    print("\nResidual Statistics:")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")

def main():
    """Main function to demonstrate regression concepts"""
    print("=== Introduction to Regression ===")
    
    # Generate and explore sample data
    df = generate_sample_data()
    
    # Visualize the relationship
    visualize_relationship(df)
    
    # Fit and visualize regression line
    model = fit_regression_line(df)
    
    # Analyze residuals
    demonstrate_residuals(df, model)
    
    print("\nAll regression examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 