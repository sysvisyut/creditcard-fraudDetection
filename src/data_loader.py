import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load the credit card fraud dataset from a CSV file."""
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    print("Data successfully loaded!\n")
    return df

def identify_features(df: pd.DataFrame):
    """Analyze distribution and print features vs target."""
    print("-" * 50)
    print("DATA EXPLORATION")
    print("-" * 50)
    
    print("\n[1] First 5 Rows:")
    print(df.head())
    
    print(f"\n[2] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n[3] Data Types:")
    print(df.dtypes)
    
    print("\n[4] Dataset Info:")
    df.info()
    
    print("\n[5] Summary Statistics:")
    print(df.describe())
    
    print("-" * 50)
    print("CLASS DISTRIBUTION & FEATURE SUMMARY")
    print("-" * 50)
    
    class_counts = df['Class'].value_counts()
    class_percentages = df['Class'].value_counts(normalize=True) * 100
    
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    pct_0 = class_percentages.get(0, 0.0)
    pct_1 = class_percentages.get(1, 0.0)
    
    print(f"\nClass 0 (Legitimate): {count_0} records ({pct_0:.2f}%)")
    print(f"Class 1 (Fraud):      {count_1} records ({pct_1:.2f}%)")
    
    target_col = "Class"
    predictor_cols = [col for col in df.columns if col != target_col]
    
    print("\nPredictor Columns:")
    print(", ".join(predictor_cols))
    
    print("\nTarget Column:")
    print(target_col)
    
    summary_box = f"""
    +-------------------------------------------------------------+
    |                      PROJECT SUMMARY                        |
    +-------------------------------------------------------------+
    | Target:      {target_col:<47}|
    | Predictors:  V1-V28, Amount, Time                           |
    +-------------------------------------------------------------+
    """
    print(summary_box)
    
    total = len(df)
    fraud_pct = (count_1 / total) * 100
    
    print(f"Fraud Rate: {fraud_pct:.2f}%")
    print(f"\nDataset contains {total:,} transactions. Only {fraud_pct:.2f}% are fraudulent — confirming severe class imbalance that requires special handling.")
    
    return predictor_cols, target_col
