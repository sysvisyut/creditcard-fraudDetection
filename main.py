import os
import warnings
import pickle
import pandas as pd
import numpy as pd_numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset from a CSV file.
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print("Data successfully loaded!\n")
    return df

def explore_data(df: pd.DataFrame):
    """
    Perform initial data exploration and display key statistics.
    """
    print("-" * 50)
    print("DATA EXPLORATION")
    print("-" * 50)
    
    # Display first 5 rows
    print("\n[1] First 5 Rows:")
    print(df.head())
    
    # Display shape
    print(f"\n[2] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display dtypes
    print("\n[3] Data Types:")
    print(df.dtypes)
    
    # Display info
    print("\n[4] Dataset Info:")
    df.info()
    
    # Display describe
    print("\n[5] Summary Statistics:")
    print(df.describe())

def analyze_classes(df: pd.DataFrame):
    """
    Analyze the distribution of the target variable (Class) and identify columns.
    """
    print("-" * 50)
    print("CLASS DISTRIBUTION & FEATURE SUMMARY")
    print("-" * 50)
    
    # Count variables
    class_counts = df['Class'].value_counts()
    class_percentages = df['Class'].value_counts(normalize=True) * 100
    
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    pct_0 = class_percentages.get(0, 0.0)
    pct_1 = class_percentages.get(1, 0.0)
    
    # Print Exact Counts and Percentages
    print(f"\nClass 0 (Legitimate): {count_0} records ({pct_0:.2f}%)")
    print(f"Class 1 (Fraud):      {count_1} records ({pct_1:.2f}%)")
    
    # Predictor and Target Identification
    target_col = "Class"
    predictor_cols = [col for col in df.columns if col != target_col]
    
    print("\nPredictor Columns:")
    print(", ".join(predictor_cols))
    
    print("\nTarget Column:")
    print(target_col)
    
    # Clear Summary Box
    summary_box = f"""
    +-------------------------------------------------------------+
    |                      PROJECT SUMMARY                        |
    +-------------------------------------------------------------+
    | Target:      Class                                          |
    | Predictors:  V1-V28, Amount, Time                           |
    +-------------------------------------------------------------+
    """
    print(summary_box)

def visualize_class_distribution(df: pd.DataFrame):
    """
    Create and save visualizations for class distribution.
    """
    print("-" * 50)
    print("VISUALIZING CLASS DISTRIBUTION")
    print("-" * 50)
    
    # Ensure plots directory exists
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # Calculate counts and percentages
    class_counts = df['Class'].value_counts()
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    total = len(df)
    fraud_pct = (count_1 / total) * 100
    
    labels = ['Legitimate (0)', 'Fraud (1)']
    counts = [count_0, count_1]
    
    # Red for fraud, blue/green for legitimate
    colors = ['#2ca02c', '#d62728'] # Green and Red
    
    # 1. Bar chart
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, palette=colors)
    plt.title('Count of Legitimate vs Fraudulent Transactions')
    plt.xlabel('Transaction Class')
    plt.ylabel('Count')
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')
                    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(colors))]
    plt.legend(handles, labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "1_bar_chart.png"))
    plt.close()
    
    # 2. Pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.2f%%', colors=colors, startangle=140)
    plt.title('Percentage Share of Fraud vs Legitimate Transactions')
    plt.legend(labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "2_pie_chart.png"))
    plt.close()
    
    # 3. Donut chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.2f%%', colors=colors, startangle=140, pctdistance=0.85)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Donut Chart: Share of Fraud vs Legitimate')
    plt.legend(labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "3_donut_chart.png"))
    plt.close()
    
    # 4. Log-scale bar chart
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, palette=colors)
    plt.yscale('log')
    plt.title('Log Scale: Legitimate vs Fraudulent Transactions')
    plt.xlabel('Transaction Class')
    plt.ylabel('Count (Log Scale)')
    
    # Annotate log bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')
                    
    plt.legend(handles, labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "4_log_scale_bar_chart.png"))
    plt.close()
    
    print(f"Fraud Rate: {fraud_pct:.2f}%")
    print(f"\nDataset contains {total:,} transactions. Only {fraud_pct:.2f}% are fraudulent — confirming severe class imbalance that requires special handling.")


def main():
    # Define dataset path relative to project root
    dataset_path = os.path.join('data', 'creditcard.csv')
    
    # Load data
    df = load_data(dataset_path)
    
    # Explore dataset
    explore_data(df)
    
    # Analyze classes and features
    analyze_classes(df)
    
    # Visualize class distribution
    visualize_class_distribution(df)

if __name__ == "__main__":
    main()
