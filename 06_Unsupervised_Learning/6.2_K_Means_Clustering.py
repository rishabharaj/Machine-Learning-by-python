"""
K-Means Clustering

This script demonstrates:
- K-Means clustering implementation
- Elbow method for choosing optimal k
- Cluster visualization
- Cluster evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def generate_data():
    """Generate sample data for K-Means clustering"""
    print("\n=== Generating Data ===")
    
    # Generate blobs data
    X, y = make_blobs(
        n_samples=500,
        centers=4,
        cluster_std=0.60,
        random_state=42
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def plot_elbow_method(X):
    """Plot elbow method for choosing optimal k"""
    print("\n=== Plotting Elbow Method ===")
    
    # Calculate inertia for different k values
    inertias = []
    k_values = range(1, 11)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png')
    plt.close()
    
    return inertias

def plot_silhouette_scores(X):
    """Plot silhouette scores for different k values"""
    print("\n=== Plotting Silhouette Scores ===")
    
    # Calculate silhouette scores
    silhouette_scores = []
    k_values = range(2, 11)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k Values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    return silhouette_scores

def perform_kmeans_clustering(X, k):
    """Perform K-Means clustering and visualize results"""
    print(f"\n=== Performing K-Means Clustering (k={k}) ===")
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.4f}")
    
    # Plot clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='x', s=200, linewidths=3)
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(f'kmeans_clustering_k{k}.png')
    plt.close()
    
    return kmeans, labels

def analyze_cluster_characteristics(X, labels, kmeans):
    """Analyze and visualize cluster characteristics"""
    print("\n=== Analyzing Cluster Characteristics ===")
    
    # Create DataFrame with cluster assignments
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    df['Cluster'] = labels
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster').agg(['mean', 'std']).round(3)
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Plot cluster distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Cluster', y='Feature1', data=df)
    plt.title('Feature 1 Distribution by Cluster')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Cluster', y='Feature2', data=df)
    plt.title('Feature 2 Distribution by Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_distributions.png')
    plt.close()

def main():
    """Main function to demonstrate K-Means clustering"""
    print("=== K-Means Clustering ===")
    
    # Generate and prepare data
    X, y = generate_data()
    
    # Plot elbow method
    inertias = plot_elbow_method(X)
    
    # Plot silhouette scores
    silhouette_scores = plot_silhouette_scores(X)
    
    # Determine optimal k (using silhouette score)
    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range started at 2
    
    # Perform K-Means clustering with optimal k
    kmeans, labels = perform_kmeans_clustering(X, optimal_k)
    
    # Analyze cluster characteristics
    analyze_cluster_characteristics(X, labels, kmeans)
    
    print("\nAll K-Means clustering examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 