"""
Introduction to Pandas

This script provides a comprehensive introduction to Pandas, covering:
- Series and DataFrame creation
- Data loading and saving
- Data inspection and cleaning
- Data selection and filtering
- Data aggregation and grouping
- Time series operations
- Data visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_series_dataframe():
    """Demonstrate creating Series and DataFrames"""
    print("\n=== Creating Series and DataFrames ===")
    
    # Create Series
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    print("Series:\n", s)
    
    # Create DataFrame from dictionary
    df = pd.DataFrame({
        'A': 1.,
        'B': pd.Timestamp('20230101'),
        'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        'D': np.array([3] * 4, dtype='int32'),
        'E': pd.Categorical(["test", "train", "test", "train"]),
        'F': 'foo'
    })
    print("\nDataFrame from dictionary:\n", df)
    
    # Create DataFrame from NumPy array
    dates = pd.date_range('20230101', periods=6)
    df2 = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print("\nDataFrame from NumPy array:\n", df2)
    
    return s, df, df2

def data_inspection():
    """Demonstrate data inspection methods"""
    print("\n=== Data Inspection ===")
    
    # Create sample DataFrame
    df = pd.DataFrame(np.random.randn(6, 4), 
                     index=pd.date_range('20230101', periods=6),
                     columns=list('ABCD'))
    
    print("DataFrame head:\n", df.head())
    print("\nDataFrame tail:\n", df.tail(3))
    print("\nDataFrame info:\n", df.info())
    print("\nDataFrame describe:\n", df.describe())
    print("\nDataFrame shape:", df.shape)
    print("\nDataFrame columns:", df.columns)
    print("\nDataFrame index:", df.index)
    
    return df

def data_selection():
    """Demonstrate data selection methods"""
    print("\n=== Data Selection ===")
    
    # Create sample DataFrame
    df = pd.DataFrame(np.random.randn(6, 4), 
                     index=pd.date_range('20230101', periods=6),
                     columns=list('ABCD'))
    
    # Column selection
    print("Select column 'A':\n", df['A'])
    print("\nSelect columns 'A' and 'B':\n", df[['A', 'B']])
    
    # Row selection
    print("\nSelect first row:\n", df.iloc[0])
    print("\nSelect rows 1 to 3:\n", df.iloc[1:4])
    
    # Label-based selection
    print("\nSelect by label:\n", df.loc['20230102':'20230104'])
    
    # Boolean indexing
    print("\nSelect where A > 0:\n", df[df['A'] > 0])
    
    return df

def data_cleaning():
    """Demonstrate data cleaning methods"""
    print("\n=== Data Cleaning ===")
    
    # Create DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    })
    print("Original DataFrame:\n", df)
    
    # Drop missing values
    print("\nDrop rows with missing values:\n", df.dropna())
    print("\nDrop columns with missing values:\n", df.dropna(axis=1))
    
    # Fill missing values
    print("\nFill missing values with 0:\n", df.fillna(0))
    print("\nFill missing values with mean:\n", df.fillna(df.mean()))
    
    # Remove duplicates
    df2 = pd.DataFrame({
        'A': [1, 2, 2, 3, 4, 4],
        'B': ['a', 'b', 'b', 'c', 'd', 'd']
    })
    print("\nOriginal DataFrame with duplicates:\n", df2)
    print("\nRemove duplicates:\n", df2.drop_duplicates())
    
    return df, df2

def data_aggregation():
    """Demonstrate data aggregation methods"""
    print("\n=== Data Aggregation ===")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': np.random.randn(8),
        'D': np.random.randn(8)
    })
    print("Original DataFrame:\n", df)
    
    # Group by
    grouped = df.groupby('A')
    print("\nGroup by 'A':\n", grouped.sum())
    
    # Multiple aggregation
    print("\nMultiple aggregation:\n", grouped.agg({'C': 'sum', 'D': 'mean'}))
    
    # Pivot tables
    print("\nPivot table:\n", pd.pivot_table(df, values='D', index=['A', 'B']))
    
    return df

def time_series_operations():
    """Demonstrate time series operations"""
    print("\n=== Time Series Operations ===")
    
    # Create time series
    rng = pd.date_range('2023-01-01', periods=100, freq='D')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print("Time series head:\n", ts.head())
    
    # Resampling
    print("\nDaily to monthly resampling:\n", ts.resample('M').mean())
    
    # Rolling window
    print("\n7-day rolling mean:\n", ts.rolling(window=7).mean().head(10))
    
    # Time zone handling
    ts_utc = ts.tz_localize('UTC')
    print("\nUTC time series:\n", ts_utc.head())
    
    return ts

def data_visualization():
    """Demonstrate data visualization with Pandas"""
    print("\n=== Data Visualization ===")
    
    # Create sample data
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('2023-01-01', periods=1000))
    ts = ts.cumsum()
    
    # Line plot
    plt.figure(figsize=(10, 6))
    ts.plot(title='Time Series')
    plt.savefig('time_series.png')
    plt.close()
    
    # Create DataFrame for more plots
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
    df = df.cumsum()
    
    # Multiple line plots
    plt.figure(figsize=(10, 6))
    df.plot()
    plt.title('Multiple Time Series')
    plt.savefig('multiple_time_series.png')
    plt.close()
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    df.plot.scatter(x='A', y='B', alpha=0.5)
    plt.title('Scatter Plot')
    plt.savefig('scatter_plot.png')
    plt.close()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    df['A'].hist(bins=30, alpha=0.5)
    plt.title('Histogram')
    plt.savefig('histogram.png')
    plt.close()
    
    return ts, df

def main():
    """Main function to demonstrate Pandas features"""
    print("=== Introduction to Pandas ===")
    
    # Create Series and DataFrames
    print("\n=== Creating Series and DataFrames ===")
    create_series_dataframe()
    
    # Data inspection
    print("\n=== Data Inspection Examples ===")
    data_inspection()
    
    # Data selection
    print("\n=== Data Selection Examples ===")
    data_selection()
    
    # Data cleaning
    print("\n=== Data Cleaning Examples ===")
    data_cleaning()
    
    # Data aggregation
    print("\n=== Data Aggregation Examples ===")
    data_aggregation()
    
    # Time series operations
    print("\n=== Time Series Operations Examples ===")
    time_series_operations()
    
    # Data visualization
    print("\n=== Data Visualization Examples ===")
    data_visualization()
    
    print("\nAll Pandas examples completed successfully!")
    print("Visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 