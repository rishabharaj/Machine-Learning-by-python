"""
Principal Component Analysis (PCA)

This script demonstrates:
- PCA implementation and visualization
- Explained variance analysis
- Dimensionality reduction effects
- Feature importance in principal components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def generate_high_dimensional_data():
    """Generate high-dimensional data for PCA demonstration"""
    print("\n=== Generating High-Dimensional Data ===")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def load_iris_data():
    """Load and prepare Iris dataset"""
    print("\n=== Loading Iris Dataset ===")
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names

def perform_pca(X, n_components=None):
    """Perform PCA and return transformed data"""
    print("\n=== Performing PCA ===")
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Print explained variance
    print("\nExplained Variance Ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")
    
    print(f"\nTotal Explained Variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca, X_pca

def plot_explained_variance(pca):
    """Plot explained variance and cumulative explained variance"""
    print("\n=== Plotting Explained Variance ===")
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('explained_variance.png')
    plt.close()

def plot_pca_components(X_pca, y, title):
    """Plot first two principal components"""
    print(f"\n=== Plotting PCA Components: {title} ===")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f'pca_components_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(pca, feature_names):
    """Plot feature importance in principal components"""
    print("\n=== Plotting Feature Importance ===")
    
    # Get feature importance for first two components
    components = pca.components_[:2]
    
    plt.figure(figsize=(12, 6))
    
    for i, component in enumerate(components):
        plt.subplot(1, 2, i+1)
        plt.barh(feature_names, component)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance in PC{i+1}')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def analyze_reconstruction_error(X, pca):
    """Analyze reconstruction error for different numbers of components"""
    print("\n=== Analyzing Reconstruction Error ===")
    
    # Calculate reconstruction error for different numbers of components
    n_features = X.shape[1]
    reconstruction_errors = []
    
    for n_components in range(1, n_features + 1):
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_pca)
        error = np.mean(np.square(X - X_reconstructed))
        reconstruction_errors.append(error)
    
    # Plot reconstruction error
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_features + 1), reconstruction_errors, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs. Number of Components')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reconstruction_error.png')
    plt.close()

def main():
    """Main function to demonstrate PCA"""
    print("=== Principal Component Analysis (PCA) ===")
    
    # Generate and analyze high-dimensional data
    X_high_dim, y_high_dim = generate_high_dimensional_data()
    pca_high_dim, X_pca_high_dim = perform_pca(X_high_dim)
    plot_explained_variance(pca_high_dim)
    plot_pca_components(X_pca_high_dim, y_high_dim, "High-Dimensional Data")
    
    # Load and analyze Iris dataset
    X_iris, y_iris, feature_names = load_iris_data()
    pca_iris, X_pca_iris = perform_pca(X_iris)
    plot_pca_components(X_pca_iris, y_iris, "Iris Dataset")
    plot_feature_importance(pca_iris, feature_names)
    
    # Analyze reconstruction error
    analyze_reconstruction_error(X_high_dim, pca_high_dim)
    
    print("\nAll PCA examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 