"""
Loading and Exploring Data

This script demonstrates:
- Loading datasets from various sources
- Basic data exploration
- Statistical analysis
- Data visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_boston

def load_datasets():
    """Load and prepare datasets"""
    print("\n=== Loading Datasets ===")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    # Load Boston Housing dataset
    boston = load_boston()
    boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
    boston_df['target'] = boston.target
    
    return iris_df, boston_df

def basic_data_exploration(df, dataset_name):
    """Perform basic data exploration"""
    print(f"\n=== Basic Data Exploration for {dataset_name} ===")
    
    # Display basic information
    print("\nDataset Info:")
    print(df.info())
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

def statistical_analysis(df, dataset_name):
    """Perform statistical analysis"""
    print(f"\n=== Statistical Analysis for {dataset_name} ===")
    
    # Calculate correlations
    print("\nCorrelation Matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Heatmap - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'correlation_heatmap_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close()

def data_visualization(df, dataset_name):
    """Create data visualizations"""
    print(f"\n=== Data Visualization for {dataset_name} ===")
    
    # Create pair plot
    if 'target' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.pairplot(df, hue='target')
        plt.suptitle(f'Pair Plot - {dataset_name}', y=1.02)
        plt.savefig(f'pair_plot_{dataset_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Create box plots for numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    if 'target' in numerical_features:
        numerical_features = numerical_features.drop('target')
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, len(numerical_features), i)
        sns.boxplot(y=df[feature])
        plt.title(feature)
    plt.suptitle(f'Box Plots - {dataset_name}', y=1.02)
    plt.tight_layout()
    plt.savefig(f'box_plots_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Create histograms
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, len(numerical_features), i)
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(feature)
    plt.suptitle(f'Histograms - {dataset_name}', y=1.02)
    plt.tight_layout()
    plt.savefig(f'histograms_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    """Main function to demonstrate data loading and exploration"""
    print("=== Data Loading and Exploration ===")
    
    # Load datasets
    iris_df, boston_df = load_datasets()
    
    # Explore Iris dataset
    basic_data_exploration(iris_df, "Iris Dataset")
    statistical_analysis(iris_df, "Iris Dataset")
    data_visualization(iris_df, "Iris Dataset")
    
    # Explore Boston dataset
    basic_data_exploration(boston_df, "Boston Housing Dataset")
    statistical_analysis(boston_df, "Boston Housing Dataset")
    data_visualization(boston_df, "Boston Housing Dataset")
    
    print("\nAll data exploration examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 