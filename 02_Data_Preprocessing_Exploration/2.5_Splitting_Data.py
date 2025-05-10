"""
Data Splitting

This script demonstrates various techniques for splitting data:
- Train-Test Split
- K-Fold Cross Validation
- Stratified K-Fold Cross Validation
- Time Series Split
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_datasets():
    """Load and prepare datasets for splitting"""
    print("\n=== Loading Datasets ===")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    # Load Boston Housing dataset
    boston = load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df['target'] = boston.target
    
    return iris_df, boston_df

def train_test_split_demo(df, target_col):
    """Demonstrate train-test split"""
    print("\n=== Train-Test Split ===")
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def kfold_cross_validation_demo(df, target_col, n_splits=5):
    """Demonstrate K-Fold Cross Validation"""
    print(f"\n=== {n_splits}-Fold Cross Validation ===")
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store scores
    scores = []
    
    # Perform cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}: R² = {score:.4f}")
    
    print(f"\nAverage R²: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")
    
    return scores

def stratified_kfold_demo(df, target_col, n_splits=5):
    """Demonstrate Stratified K-Fold Cross Validation"""
    print(f"\n=== Stratified {n_splits}-Fold Cross Validation ===")
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store scores
    scores = []
    
    # Perform cross validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}: R² = {score:.4f}")
    
    print(f"\nAverage R²: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")
    
    return scores

def time_series_split_demo(df, target_col, n_splits=5):
    """Demonstrate Time Series Split"""
    print(f"\n=== Time Series Split ({n_splits} splits) ===")
    
    # Sort data by index (assuming it's time-based)
    df = df.sort_index()
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store scores
    scores = []
    
    # Perform time series split
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate model
        score = model.score(X_val, y_val)
        scores.append(score)
        
        print(f"Fold {fold + 1}:")
        print(f"  Training set size: {len(X_train)} samples")
        print(f"  Validation set size: {len(X_val)} samples")
        print(f"  R² = {score:.4f}")
    
    print(f"\nAverage R²: {np.mean(scores):.4f}")
    print(f"Standard Deviation: {np.std(scores):.4f}")
    
    return scores

def plot_scores_comparison(scores_dict, dataset_name):
    """Plot comparison of different splitting methods"""
    print(f"\n=== Plotting Scores Comparison for {dataset_name} ===")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot scores for each method
    for method, scores in scores_dict.items():
        plt.plot(range(1, len(scores) + 1), scores, marker='o', label=method)
    
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title(f'Cross Validation Scores Comparison ({dataset_name})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'splitting_comparison_{dataset_name}.png')
    plt.close()

def main():
    """Main function to demonstrate data splitting"""
    print("=== Data Splitting ===")
    
    # Load datasets
    iris_df, boston_df = load_datasets()
    
    # Process each dataset
    for df, name, target_col in [
        (iris_df, 'iris', 'target'),
        (boston_df, 'boston', 'target')
    ]:
        print(f"\nProcessing {name} dataset...")
        
        # Store scores for comparison
        scores_dict = {}
        
        # Perform different splitting methods
        if name == 'iris':
            # For classification dataset
            scores_dict['Stratified K-Fold'] = stratified_kfold_demo(df, target_col)
        else:
            # For regression dataset
            scores_dict['K-Fold'] = kfold_cross_validation_demo(df, target_col)
            scores_dict['Time Series'] = time_series_split_demo(df, target_col)
        
        # Plot comparison
        plot_scores_comparison(scores_dict, name)
        
        # Demonstrate train-test split
        X_train, X_test, y_train, y_test = train_test_split_demo(df, target_col)
    
    print("\nAll splitting examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 