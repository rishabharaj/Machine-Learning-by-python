"""
Introduction to Unsupervised Learning

This script demonstrates:
- Basic concepts of unsupervised learning
- Clustering visualization
- Dimensionality reduction visualization
- Understanding patterns in unlabeled data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def generate_sample_data():
    """Generate different types of sample data for clustering"""
    print("\n=== Generating Sample Data ===")
    
    # Generate blobs data
    X_blobs, y_blobs = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.60,
        random_state=42
    )
    
    # Generate moons data
    X_moons, y_moons = make_moons(
        n_samples=300,
        noise=0.05,
        random_state=42
    )
    
    # Generate circles data
    X_circles, y_circles = make_circles(
        n_samples=300,
        noise=0.05,
        factor=0.5,
        random_state=42
    )
    
    return {
        'blobs': (X_blobs, y_blobs),
        'moons': (X_moons, y_moons),
        'circles': (X_circles, y_circles)
    }

def visualize_clustering_data(data_dict):
    """Visualize different types of clustering data"""
    print("\n=== Visualizing Clustering Data ===")
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, (X, y)) in enumerate(data_dict.items(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
        plt.title(f'{name.capitalize()} Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('clustering_data_types.png')
    plt.close()

def demonstrate_clustering(data_dict):
    """Demonstrate different clustering algorithms"""
    print("\n=== Demonstrating Clustering Algorithms ===")
    
    # Initialize clustering algorithms
    kmeans = KMeans(n_clusters=4, random_state=42)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, (X, y)) in enumerate(data_dict.items(), 1):
        # KMeans clustering
        plt.subplot(3, 2, 2*i-1)
        kmeans.fit(X)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
        plt.title(f'KMeans on {name.capitalize()} Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # DBSCAN clustering
        plt.subplot(3, 2, 2*i)
        dbscan.fit(X)
        plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis', s=50)
        plt.title(f'DBSCAN on {name.capitalize()} Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('clustering_algorithms.png')
    plt.close()

def demonstrate_dimensionality_reduction():
    """Demonstrate dimensionality reduction techniques"""
    print("\n=== Demonstrating Dimensionality Reduction ===")
    
    # Generate high-dimensional data
    X, _ = make_blobs(
        n_samples=300,
        n_features=10,
        centers=4,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis', s=50)
    plt.title('PCA (2D Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='viridis', s=50)
    plt.title('t-SNE (2D Projection)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('dimensionality_reduction.png')
    plt.close()
    
    # Print explained variance ratio for PCA
    print("\nPCA Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

def main():
    """Main function to demonstrate unsupervised learning concepts"""
    print("=== Introduction to Unsupervised Learning ===")
    
    # Generate and visualize sample data
    data_dict = generate_sample_data()
    visualize_clustering_data(data_dict)
    
    # Demonstrate clustering algorithms
    demonstrate_clustering(data_dict)
    
    # Demonstrate dimensionality reduction
    demonstrate_dimensionality_reduction()
    
    print("\nAll unsupervised learning examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 