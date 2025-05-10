"""
Introduction to Data Visualization

This script provides a comprehensive introduction to data visualization using Matplotlib and Seaborn, covering:
- Basic plotting with Matplotlib
- Advanced plotting with Seaborn
- Statistical visualizations
- Customizing plots
- Saving and exporting visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_matplotlib_plots():
    """Demonstrate basic Matplotlib plotting"""
    print("\n=== Basic Matplotlib Plots ===")
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='sin(x)')
    plt.title('Basic Line Plot')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.legend()
    plt.savefig('basic_line_plot.png')
    plt.close()
    
    # Scatter plot
    x2 = np.random.rand(50)
    y2 = np.random.rand(50)
    plt.figure(figsize=(10, 6))
    plt.scatter(x2, y2, c='red', alpha=0.5)
    plt.title('Scatter Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('scatter_plot.png')
    plt.close()
    
    # Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [10, 20, 15, 25]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='skyblue')
    plt.title('Bar Plot')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.savefig('bar_plot.png')
    plt.close()
    
    return x, y, x2, y2, categories, values

def advanced_matplotlib_plots():
    """Demonstrate advanced Matplotlib features"""
    print("\n=== Advanced Matplotlib Plots ===")
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Multiple plots
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', linestyle='-', color='blue')
    plt.plot(x, y2, label='cos(x)', linestyle='--', color='red')
    plt.title('Multiple Line Plots')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.savefig('multiple_line_plots.png')
    plt.close()
    
    # Subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(x, y1, color='blue')
    ax1.set_title('Sin(x)')
    ax2.plot(x, y2, color='red')
    ax2.set_title('Cos(x)')
    plt.tight_layout()
    plt.savefig('subplots.png')
    plt.close()
    
    # Histogram
    data = np.random.normal(0, 1, 1000)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7, color='green')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')
    plt.close()
    
    return x, y1, y2, data

def seaborn_plots():
    """Demonstrate Seaborn plotting"""
    print("\n=== Seaborn Plots ===")
    
    # Load sample dataset
    tips = sns.load_dataset('tips')
    
    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x='total_bill', y='tip', data=tips)
    plt.title('Scatter Plot with Regression Line')
    plt.savefig('seaborn_regplot.png')
    plt.close()
    
    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='day', y='total_bill', data=tips)
    plt.title('Box Plot')
    plt.savefig('seaborn_boxplot.png')
    plt.close()
    
    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='day', y='total_bill', data=tips)
    plt.title('Violin Plot')
    plt.savefig('seaborn_violinplot.png')
    plt.close()
    
    # Heatmap
    correlation = tips.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('seaborn_heatmap.png')
    plt.close()
    
    return tips

def statistical_visualizations():
    """Demonstrate statistical visualizations"""
    print("\n=== Statistical Visualizations ===")
    
    # Load sample dataset
    iris = sns.load_dataset('iris')
    
    # Pair plot
    plt.figure(figsize=(10, 6))
    sns.pairplot(iris, hue='species')
    plt.savefig('pair_plot.png')
    plt.close()
    
    # Joint plot
    plt.figure(figsize=(10, 6))
    sns.jointplot(x='sepal_length', y='sepal_width', data=iris, kind='scatter')
    plt.savefig('joint_plot.png')
    plt.close()
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.distplot(iris['sepal_length'], kde=True)
    plt.title('Distribution Plot')
    plt.savefig('distribution_plot.png')
    plt.close()
    
    return iris

def customizing_plots():
    """Demonstrate plot customization"""
    print("\n=== Customizing Plots ===")
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Customized plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 
             color='purple',
             linewidth=2,
             linestyle='--',
             marker='o',
             markersize=5,
             markerfacecolor='red',
             markeredgecolor='black',
             markeredgewidth=1)
    
    # Customize axes
    plt.title('Customized Plot', fontsize=16, fontweight='bold')
    plt.xlabel('x-axis', fontsize=14)
    plt.ylabel('y-axis', fontsize=14)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add text
    plt.text(5, 0.5, 'Maximum Point', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add legend
    plt.legend(['sin(x)'], fontsize=12)
    
    # Save with high DPI
    plt.savefig('customized_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return x, y

def main():
    """Main function to demonstrate data visualization features"""
    print("=== Introduction to Data Visualization ===")
    
    # Basic Matplotlib plots
    print("\n=== Basic Matplotlib Examples ===")
    basic_matplotlib_plots()
    
    # Advanced Matplotlib plots
    print("\n=== Advanced Matplotlib Examples ===")
    advanced_matplotlib_plots()
    
    # Seaborn plots
    print("\n=== Seaborn Examples ===")
    seaborn_plots()
    
    # Statistical visualizations
    print("\n=== Statistical Visualization Examples ===")
    statistical_visualizations()
    
    # Customizing plots
    print("\n=== Plot Customization Examples ===")
    customizing_plots()
    
    print("\nAll visualization examples completed successfully!")
    print("Visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 