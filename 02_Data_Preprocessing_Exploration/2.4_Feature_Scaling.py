"""
Feature Scaling

This script demonstrates various techniques for feature scaling:
- Standardization (Z-score normalization)
- Min-Max Scaling
- Robust Scaling
- Normalization (L1, L2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.datasets import load_iris, load_boston

def load_datasets():
    """Load and prepare datasets for scaling"""
    print("\n=== Loading Datasets ===")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Load Boston Housing dataset
    boston = load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    
    return iris_df, boston_df

def standardize_features(df):
    """Demonstrate standardization (Z-score normalization)"""
    print("\n=== Standardization (Z-score) ===")
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Apply standardization
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    print("\nOriginal statistics:")
    print(df.describe())
    print("\nStandardized statistics:")
    print(scaled_df.describe())
    
    return scaled_df

def minmax_scale_features(df):
    """Demonstrate min-max scaling"""
    print("\n=== Min-Max Scaling ===")
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Apply min-max scaling
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    print("\nOriginal statistics:")
    print(df.describe())
    print("\nMin-Max scaled statistics:")
    print(scaled_df.describe())
    
    return scaled_df

def robust_scale_features(df):
    """Demonstrate robust scaling"""
    print("\n=== Robust Scaling ===")
    
    # Initialize scaler
    scaler = RobustScaler()
    
    # Apply robust scaling
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    print("\nOriginal statistics:")
    print(df.describe())
    print("\nRobust scaled statistics:")
    print(scaled_df.describe())
    
    return scaled_df

def normalize_features(df):
    """Demonstrate normalization (L1 and L2)"""
    print("\n=== Normalization (L1 and L2) ===")
    
    # Initialize normalizers
    l1_normalizer = Normalizer(norm='l1')
    l2_normalizer = Normalizer(norm='l2')
    
    # Apply normalization
    l1_scaled_data = l1_normalizer.fit_transform(df)
    l2_scaled_data = l2_normalizer.fit_transform(df)
    
    l1_scaled_df = pd.DataFrame(l1_scaled_data, columns=df.columns)
    l2_scaled_df = pd.DataFrame(l2_scaled_data, columns=df.columns)
    
    print("\nL1 normalized statistics:")
    print(l1_scaled_df.describe())
    print("\nL2 normalized statistics:")
    print(l2_scaled_df.describe())
    
    return l1_scaled_df, l2_scaled_df

def plot_scaling_comparison(df, scaled_dfs, dataset_name):
    """Plot comparison of different scaling methods"""
    print(f"\n=== Plotting Scaling Comparison for {dataset_name} ===")
    
    # Select a feature for visualization
    feature = df.columns[0]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original data
    plt.subplot(2, 3, 1)
    sns.histplot(df[feature], kde=True)
    plt.title('Original Data')
    
    # Standardized data
    plt.subplot(2, 3, 2)
    sns.histplot(scaled_dfs['standardized'][feature], kde=True)
    plt.title('Standardized')
    
    # Min-Max scaled data
    plt.subplot(2, 3, 3)
    sns.histplot(scaled_dfs['minmax'][feature], kde=True)
    plt.title('Min-Max Scaled')
    
    # Robust scaled data
    plt.subplot(2, 3, 4)
    sns.histplot(scaled_dfs['robust'][feature], kde=True)
    plt.title('Robust Scaled')
    
    # L1 normalized data
    plt.subplot(2, 3, 5)
    sns.histplot(scaled_dfs['l1'][feature], kde=True)
    plt.title('L1 Normalized')
    
    # L2 normalized data
    plt.subplot(2, 3, 6)
    sns.histplot(scaled_dfs['l2'][feature], kde=True)
    plt.title('L2 Normalized')
    
    plt.tight_layout()
    plt.savefig(f'scaling_comparison_{dataset_name}.png')
    plt.close()

def main():
    """Main function to demonstrate feature scaling"""
    print("=== Feature Scaling ===")
    
    # Load datasets
    iris_df, boston_df = load_datasets()
    
    # Process each dataset
    for df, name in [(iris_df, 'iris'), (boston_df, 'boston')]:
        print(f"\nProcessing {name} dataset...")
        
        # Apply different scaling methods
        standardized_df = standardize_features(df)
        minmax_df = minmax_scale_features(df)
        robust_df = robust_scale_features(df)
        l1_df, l2_df = normalize_features(df)
        
        # Store scaled DataFrames
        scaled_dfs = {
            'standardized': standardized_df,
            'minmax': minmax_df,
            'robust': robust_df,
            'l1': l1_df,
            'l2': l2_df
        }
        
        # Plot comparison
        plot_scaling_comparison(df, scaled_dfs, name)
    
    print("\nAll scaling examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 