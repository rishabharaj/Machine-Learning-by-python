"""
Encoding Categorical Data

This script demonstrates various techniques for encoding categorical data:
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding
- Target Encoding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
import seaborn as sns

def create_sample_data():
    """Create sample dataset with categorical features"""
    print("\n=== Creating Sample Dataset ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data
    data = {
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], 100),
        'experience_level': np.random.choice(['Junior', 'Mid', 'Senior', 'Lead'], 100),
        'salary': np.random.normal(50000, 15000, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def label_encoding(df):
    """Demonstrate label encoding"""
    print("\n=== Label Encoding ===")
    
    # Create copy of DataFrame
    df_label = df.copy()
    
    # Initialize label encoder
    label_encoder = LabelEncoder()
    
    # Apply label encoding to each categorical column
    for col in ['education', 'city', 'experience_level']:
        df_label[f'{col}_label'] = label_encoder.fit_transform(df[col])
        print(f"\n{col} mapping:")
        for i, category in enumerate(label_encoder.classes_):
            print(f"{category}: {i}")
    
    return df_label

def one_hot_encoding(df):
    """Demonstrate one-hot encoding"""
    print("\n=== One-Hot Encoding ===")
    
    # Create copy of DataFrame
    df_onehot = df.copy()
    
    # Apply one-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_features = onehot_encoder.fit_transform(df[['education', 'city', 'experience_level']])
    
    # Get feature names
    feature_names = []
    for i, col in enumerate(['education', 'city', 'experience_level']):
        for category in df[col].unique():
            feature_names.append(f"{col}_{category}")
    
    # Create DataFrame with one-hot encoded features
    df_onehot_encoded = pd.DataFrame(encoded_features, columns=feature_names)
    
    print("\nOne-hot encoded features:")
    print(df_onehot_encoded.head())
    
    return df_onehot_encoded

def ordinal_encoding(df):
    """Demonstrate ordinal encoding"""
    print("\n=== Ordinal Encoding ===")
    
    # Create copy of DataFrame
    df_ordinal = df.copy()
    
    # Define categories order
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
    experience_order = ['Junior', 'Mid', 'Senior', 'Lead']
    
    # Initialize ordinal encoder
    ordinal_encoder = OrdinalEncoder(
        categories=[education_order, df['city'].unique(), experience_order]
    )
    
    # Apply ordinal encoding
    encoded_features = ordinal_encoder.fit_transform(df[['education', 'city', 'experience_level']])
    
    # Create DataFrame with ordinal encoded features
    df_ordinal_encoded = pd.DataFrame(
        encoded_features,
        columns=['education_ordinal', 'city_ordinal', 'experience_ordinal']
    )
    
    print("\nOrdinal encoded features:")
    print(df_ordinal_encoded.head())
    
    return df_ordinal_encoded

def target_encoding(df):
    """Demonstrate target encoding"""
    print("\n=== Target Encoding ===")
    
    # Create copy of DataFrame
    df_target = df.copy()
    
    # Initialize target encoder
    target_encoder = TargetEncoder()
    
    # Apply target encoding
    encoded_features = target_encoder.fit_transform(
        df[['education', 'city', 'experience_level']],
        df['salary']
    )
    
    # Create DataFrame with target encoded features
    df_target_encoded = pd.DataFrame(
        encoded_features,
        columns=['education_target', 'city_target', 'experience_target']
    )
    
    print("\nTarget encoded features:")
    print(df_target_encoded.head())
    
    return df_target_encoded

def compare_encoding_methods(df, df_label, df_onehot, df_ordinal, df_target):
    """Compare different encoding methods"""
    print("\n=== Comparing Encoding Methods ===")
    
    # Select a categorical column for comparison
    col = 'education'
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Original data
    plt.subplot(2, 2, 1)
    sns.boxplot(x=col, y='salary', data=df)
    plt.title('Original Data')
    plt.xticks(rotation=45)
    
    # Label encoding
    plt.subplot(2, 2, 2)
    sns.boxplot(x=f'{col}_label', y='salary', data=df_label)
    plt.title('Label Encoding')
    
    # One-hot encoding
    plt.subplot(2, 2, 3)
    onehot_cols = [c for c in df_onehot.columns if c.startswith(f'{col}_')]
    for i, onehot_col in enumerate(onehot_cols):
        sns.boxplot(x=onehot_col, y='salary', data=pd.concat([df_onehot[onehot_col], df['salary']], axis=1))
    plt.title('One-Hot Encoding')
    
    # Target encoding
    plt.subplot(2, 2, 4)
    sns.boxplot(x=f'{col}_target', y='salary', data=pd.concat([df_target[f'{col}_target'], df['salary']], axis=1))
    plt.title('Target Encoding')
    
    plt.tight_layout()
    plt.savefig('encoding_comparison.png')
    plt.close()

def main():
    """Main function to demonstrate categorical data encoding"""
    print("=== Categorical Data Encoding ===")
    
    # Create sample data
    df = create_sample_data()
    
    # Demonstrate different encoding methods
    df_label = label_encoding(df)
    df_onehot = one_hot_encoding(df)
    df_ordinal = ordinal_encoding(df)
    df_target = target_encoding(df)
    
    # Compare encoding methods
    compare_encoding_methods(df, df_label, df_onehot, df_ordinal, df_target)
    
    print("\nAll encoding examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 