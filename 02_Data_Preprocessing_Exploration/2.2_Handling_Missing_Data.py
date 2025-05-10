"""
Handling Missing Data

This script demonstrates various techniques for handling missing data:
- Identifying missing values
- Dropping missing values
- Simple imputation (mean, median, mode)
- Advanced imputation (KNN, Iterative)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

def create_sample_data():
    """Create sample dataset with missing values"""
    print("\n=== Creating Sample Dataset ===")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data
    data = {
        'age': np.random.normal(35, 10, 100),
        'income': np.random.normal(50000, 15000, 100),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'experience': np.random.normal(10, 5, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[df.sample(frac=0.1).index, 'age'] = np.nan
    df.loc[df.sample(frac=0.15).index, 'income'] = np.nan
    df.loc[df.sample(frac=0.2).index, 'education'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'experience'] = np.nan
    
    return df

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("\n=== Analyzing Missing Data ===")
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create missing data summary
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    
    print("\nMissing Data Summary:")
    print(missing_summary)
    
    # Plot missing data
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Pattern')
    plt.savefig('missing_data_pattern.png')
    plt.close()

def drop_missing_values(df):
    """Demonstrate dropping missing values"""
    print("\n=== Dropping Missing Values ===")
    
    # Drop rows with any missing values
    df_dropped = df.dropna()
    print(f"Original shape: {df.shape}")
    print(f"Shape after dropping: {df_dropped.shape}")
    
    return df_dropped

def simple_imputation(df):
    """Demonstrate simple imputation techniques"""
    print("\n=== Simple Imputation ===")
    
    # Create copy of DataFrame
    df_imputed = df.copy()
    
    # Mean imputation for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    imputer_mean = SimpleImputer(strategy='mean')
    df_imputed[numerical_cols] = imputer_mean.fit_transform(df[numerical_cols])
    
    # Mode imputation for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    imputer_mode = SimpleImputer(strategy='most_frequent')
    df_imputed[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])
    
    print("\nImputation Summary:")
    print("Mean values used:", dict(zip(numerical_cols, imputer_mean.statistics_)))
    print("Mode values used:", dict(zip(categorical_cols, imputer_mode.statistics_)))
    
    return df_imputed

def knn_imputation(df):
    """Demonstrate KNN imputation"""
    print("\n=== KNN Imputation ===")
    
    # Create copy of DataFrame
    df_knn = df.copy()
    
    # Convert categorical to numerical for KNN
    df_knn['education'] = pd.Categorical(df_knn['education']).codes
    
    # Perform KNN imputation
    imputer_knn = KNNImputer(n_neighbors=5)
    df_knn_imputed = pd.DataFrame(
        imputer_knn.fit_transform(df_knn),
        columns=df_knn.columns
    )
    
    # Convert education back to categorical
    df_knn_imputed['education'] = pd.Categorical.from_codes(
        df_knn_imputed['education'].astype(int),
        categories=pd.Categorical(df['education']).categories
    )
    
    return df_knn_imputed

def iterative_imputation(df):
    """Demonstrate iterative imputation"""
    print("\n=== Iterative Imputation ===")
    
    # Create copy of DataFrame
    df_iter = df.copy()
    
    # Convert categorical to numerical for imputation
    df_iter['education'] = pd.Categorical(df_iter['education']).codes
    
    # Perform iterative imputation
    imputer_iter = IterativeImputer(
        estimator=RandomForestRegressor(),
        random_state=42
    )
    df_iter_imputed = pd.DataFrame(
        imputer_iter.fit_transform(df_iter),
        columns=df_iter.columns
    )
    
    # Convert education back to categorical
    df_iter_imputed['education'] = pd.Categorical.from_codes(
        df_iter_imputed['education'].astype(int),
        categories=pd.Categorical(df['education']).categories
    )
    
    return df_iter_imputed

def compare_imputation_methods(df, df_simple, df_knn, df_iter):
    """Compare different imputation methods"""
    print("\n=== Comparing Imputation Methods ===")
    
    # Select a numerical column for comparison
    col = 'income'
    
    # Create comparison plot
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 4, 1)
    sns.histplot(data=df, x=col, kde=True)
    plt.title('Original Data')
    
    # Simple imputation
    plt.subplot(1, 4, 2)
    sns.histplot(data=df_simple, x=col, kde=True)
    plt.title('Simple Imputation')
    
    # KNN imputation
    plt.subplot(1, 4, 3)
    sns.histplot(data=df_knn, x=col, kde=True)
    plt.title('KNN Imputation')
    
    # Iterative imputation
    plt.subplot(1, 4, 4)
    sns.histplot(data=df_iter, x=col, kde=True)
    plt.title('Iterative Imputation')
    
    plt.tight_layout()
    plt.savefig('imputation_comparison.png')
    plt.close()

def main():
    """Main function to demonstrate missing data handling"""
    print("=== Handling Missing Data ===")
    
    # Create sample data
    df = create_sample_data()
    
    # Analyze missing data
    analyze_missing_data(df)
    
    # Demonstrate different imputation methods
    df_dropped = drop_missing_values(df)
    df_simple = simple_imputation(df)
    df_knn = knn_imputation(df)
    df_iter = iterative_imputation(df)
    
    # Compare imputation methods
    compare_imputation_methods(df, df_simple, df_knn, df_iter)
    
    print("\nAll missing data handling examples completed successfully!")
    print("Results have been saved as PNG files.")

if __name__ == "__main__":
    main() 