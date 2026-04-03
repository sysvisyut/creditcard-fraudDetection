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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Data Preprocessing: Missing values, outliers, and noise.
    """
    print("-" * 50)
    print("DATA PREPROCESSING")
    print("-" * 50)
    
    # Ensure plots directory exists
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # ---------- MISSING DATA ----------
    print("\n--- MISSING DATA ---")
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()
    
    # 2. Print a heatmap of missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "missing_heatmap.png"))
    plt.close()
    
    # 3. Impute using median for numerical columns
    if total_missing_before > 0:
        for col in df.columns:
            if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                
    missing_after = df.isnull().sum()
    total_missing_after = missing_after.sum()
    
    # 4. Print statement
    print(f"Missing values before: {total_missing_before} | After imputation: {total_missing_after}")

    # ---------- OUTLIER HANDLING ----------
    print("\n--- OUTLIER HANDLING ---")
    cols_to_clip = ['Amount', 'Time']
    
    # Save before data for plotting
    df_before_outliers_amount = df['Amount'].copy()
    df_before_outliers_time = df['Time'].copy()
    
    outliers_clipped_dict = {}
    
    for col in cols_to_clip:
        # 1. & 2. Calculate IQR and print count
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"Outliers detected in {col} (IQR method): {outliers_count}")
        
        # 3. Apply Winsorization (1st and 99th percentile)
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        
        # Track how many points actually get clipped for the summary table
        clipped_count = ((df[col] < p1) | (df[col] > p99)).sum()
        outliers_clipped_dict[col] = clipped_count
        
        df[col] = df[col].clip(lower=p1, upper=p99)

    # 4. Plot boxplots Before vs After
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(y=df_before_outliers_amount, ax=axes[0, 0], color='lightblue').set_title('Amount (Before)')
    sns.boxplot(y=df['Amount'], ax=axes[0, 1], color='lightgreen').set_title('Amount (After Winsorization)')
    sns.boxplot(y=df_before_outliers_time, ax=axes[1, 0], color='lightblue').set_title('Time (Before)')
    sns.boxplot(y=df['Time'], ax=axes[1, 1], color='lightgreen').set_title('Time (After Winsorization)')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "outliers_before_after.png"))
    plt.close()

    # ---------- NOISE HANDLING ----------
    print("\n--- NOISE HANDLING ---")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicate_count}")
    
    shape_before = df.shape
    if duplicate_count > 0:
        df = df.drop_duplicates()
    shape_after = df.shape
    
    print(f"Shape before dropping duplicates: {shape_before}")
    print(f"Shape after dropping duplicates: {shape_after}")

    # ---------- SUMMARY TABLE ----------
    print("\n--- PREPROCESSING SUMMARY TABLE ---")
    summary_data = []
    
    for i, col in enumerate(df.columns):
        mb = missing_before[col]
        ma = missing_after[col]
        
        if col in cols_to_clip:
            out_clipped = outliers_clipped_dict[col]
        else:
            out_clipped = "N/A"
            
        dups = duplicate_count if i == 0 else "-"
        
        summary_data.append([col, mb, ma, out_clipped, dups])
        
    summary_df = pd.DataFrame(summary_data, columns=['column', 'missing_before', 'missing_after', 'outliers_clipped', 'duplicates_removed'])
    print(summary_df.to_string(index=False))

    return df


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
    
    # Data Preprocessing
    df = preprocess_data(df)

if __name__ == "__main__":
    main()
