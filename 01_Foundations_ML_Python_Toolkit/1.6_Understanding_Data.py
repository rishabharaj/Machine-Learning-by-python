"""
Lesson 1.6: Understanding Data

This lesson covers:
- Loading and exploring datasets
- Understanding features and labels
- Data types and distributions
- Basic statistical analysis
- Data visualization for understanding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_and_explore_iris():
    """Load and explore the Iris dataset"""
    # Load the dataset
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data['species'] = data['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Basic information
    print("Dataset shape:", data.shape)
    print("\nFirst few rows:\n", data.head())
    print("\nData types:\n", data.dtypes)
    print("\nBasic statistics:\n", data.describe())
    
    return data

def analyze_features(data):
    """Analyze features of the dataset"""
    # Feature distributions
    print("\n=== Feature Distributions ===")
    for feature in data.columns[:-2]:  # Exclude target and species
        print(f"\n{feature} distribution:")
        print("Mean:", data[feature].mean())
        print("Median:", data[feature].median())
        print("Standard deviation:", data[feature].std())
        print("Range:", data[feature].max() - data[feature].min())
    
    # Correlation between features
    print("\n=== Feature Correlations ===")
    correlation = data.iloc[:, :-2].corr()
    print(correlation)
    
    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.savefig('feature_correlations.png')
    plt.close()

def analyze_target(data):
    """Analyze the target variable"""
    # Target distribution
    print("\n=== Target Distribution ===")
    target_counts = data['species'].value_counts()
    print(target_counts)
    
    # Visualize target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='species')
    plt.title('Species Distribution')
    plt.savefig('target_distribution.png')
    plt.close()

def visualize_features(data):
    """Visualize features and their relationships"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature distributions by species
    features = data.columns[:-2]
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        sns.boxplot(data=data, x='species', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} by Species')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Pair plot
    plt.figure(figsize=(12, 10))
    sns.pairplot(data, hue='species', markers=['o', 's', 'D'])
    plt.savefig('pair_plot.png')
    plt.close()

def feature_importance(data):
    """Analyze feature importance"""
    # Calculate feature importance using correlation with target
    feature_importance = {}
    for feature in data.columns[:-2]:
        correlation = data[feature].corr(data['target'])
        feature_importance[feature] = abs(correlation)
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    print("\n=== Feature Importance ===")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    features, importance = zip(*sorted_features)
    plt.bar(features, importance)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    """Main function to demonstrate data understanding"""
    # Load and explore the dataset
    print("=== Loading and Exploring Dataset ===")
    data = load_and_explore_iris()
    
    # Analyze features
    print("\n=== Analyzing Features ===")
    analyze_features(data)
    
    # Analyze target
    print("\n=== Analyzing Target ===")
    analyze_target(data)
    
    # Visualize features
    print("\n=== Visualizing Features ===")
    visualize_features(data)
    
    # Analyze feature importance
    print("\n=== Analyzing Feature Importance ===")
    feature_importance(data)
    
    print("\nAll visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 