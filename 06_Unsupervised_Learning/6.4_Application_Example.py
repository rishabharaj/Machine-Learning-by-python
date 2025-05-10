"""
Unsupervised Learning Application Example

This script demonstrates:
- Customer segmentation using clustering
- Dimensionality reduction for visualization
- Feature analysis and interpretation
- Practical application of unsupervised learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def generate_customer_data():
    """Generate synthetic customer data"""
    print("\n=== Generating Customer Data ===")
    
    # Generate synthetic customer data
    n_samples = 1000
    n_features = 10
    
    # Generate clusters with different characteristics
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=4,
        cluster_std=0.8,
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'Annual_Income',
        'Age',
        'Spending_Score',
        'Purchase_Frequency',
        'Average_Transaction_Value',
        'Customer_Tenure',
        'Product_Diversity',
        'Online_Activity',
        'Loyalty_Points',
        'Customer_Satisfaction'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some realistic transformations
    df['Annual_Income'] = df['Annual_Income'] * 10000 + 50000
    df['Age'] = df['Age'] * 10 + 30
    df['Spending_Score'] = (df['Spending_Score'] * 20 + 50).clip(0, 100)
    df['Purchase_Frequency'] = (df['Purchase_Frequency'] * 5 + 10).clip(0, 30)
    df['Average_Transaction_Value'] = df['Average_Transaction_Value'] * 100 + 200
    df['Customer_Tenure'] = (df['Customer_Tenure'] * 5 + 2).clip(0, 10)
    df['Product_Diversity'] = (df['Product_Diversity'] * 3 + 5).clip(0, 10)
    df['Online_Activity'] = (df['Online_Activity'] * 5 + 5).clip(0, 10)
    df['Loyalty_Points'] = df['Loyalty_Points'] * 1000 + 5000
    df['Customer_Satisfaction'] = (df['Customer_Satisfaction'] * 2 + 3).clip(1, 5)
    
    return df

def preprocess_data(df):
    """Preprocess customer data"""
    print("\n=== Preprocessing Data ===")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return X_scaled

def perform_customer_segmentation(X):
    """Perform customer segmentation using K-Means"""
    print("\n=== Performing Customer Segmentation ===")
    
    # Find optimal number of clusters
    silhouette_scores = []
    k_values = range(2, 11)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    # Choose optimal k
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    return kmeans, labels, silhouette_scores

def visualize_segments(X, labels, kmeans):
    """Visualize customer segments"""
    print("\n=== Visualizing Customer Segments ===")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot segments
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='x', s=200, linewidths=3)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Customer Segments')
    
    plt.tight_layout()
    plt.savefig('customer_segments.png')
    plt.close()

def analyze_segment_characteristics(df, labels):
    """Analyze characteristics of each customer segment"""
    print("\n=== Analyzing Segment Characteristics ===")
    
    # Add cluster labels to DataFrame
    df['Segment'] = labels
    
    # Calculate segment statistics
    segment_stats = df.groupby('Segment').agg(['mean', 'std']).round(2)
    
    # Print segment characteristics
    print("\nSegment Characteristics:")
    print(segment_stats)
    
    # Plot segment characteristics
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(df.columns[:-1], 1):
        plt.subplot(3, 4, i)
        sns.boxplot(x='Segment', y=feature, data=df)
        plt.title(f'{feature} by Segment')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('segment_characteristics.png')
    plt.close()
    
    return segment_stats

def create_segment_profiles(segment_stats):
    """Create customer segment profiles"""
    print("\n=== Creating Segment Profiles ===")
    
    profiles = {}
    for segment in segment_stats.index:
        profile = {
            'Size': len(df[df['Segment'] == segment]),
            'Characteristics': {}
        }
        
        for feature in segment_stats.columns.levels[0]:
            mean = segment_stats.loc[segment, (feature, 'mean')]
            std = segment_stats.loc[segment, (feature, 'std')]
            
            if feature == 'Annual_Income':
                profile['Characteristics'][feature] = f"${mean:,.0f} ± ${std:,.0f}"
            elif feature in ['Age', 'Customer_Tenure']:
                profile['Characteristics'][feature] = f"{mean:.1f} ± {std:.1f} years"
            elif feature in ['Spending_Score', 'Customer_Satisfaction']:
                profile['Characteristics'][feature] = f"{mean:.1f} ± {std:.1f} (1-100)"
            else:
                profile['Characteristics'][feature] = f"{mean:.1f} ± {std:.1f}"
        
        profiles[f'Segment {segment}'] = profile
    
    # Print segment profiles
    print("\nSegment Profiles:")
    for segment, profile in profiles.items():
        print(f"\n{segment}:")
        print(f"Size: {profile['Size']} customers")
        print("Characteristics:")
        for feature, value in profile['Characteristics'].items():
            print(f"  - {feature}: {value}")
    
    return profiles

def main():
    """Main function to demonstrate customer segmentation"""
    print("=== Customer Segmentation Example ===")
    
    # Generate and preprocess data
    df = generate_customer_data()
    X = preprocess_data(df)
    
    # Perform customer segmentation
    kmeans, labels, silhouette_scores = perform_customer_segmentation(X)
    
    # Visualize segments
    visualize_segments(X, labels, kmeans)
    
    # Analyze segment characteristics
    segment_stats = analyze_segment_characteristics(df, labels)
    
    # Create segment profiles
    profiles = create_segment_profiles(segment_stats)
    
    print("\nAll customer segmentation examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 