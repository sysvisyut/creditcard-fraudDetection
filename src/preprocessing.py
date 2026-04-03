import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

def check_missing(df: pd.DataFrame, return_metrics=False):
    print("\n--- MISSING DATA ---")
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "missing_heatmap.png"))
    plt.close()
    
    if total_missing_before > 0:
        for col in df.columns:
            if df[col].isnull().any() and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                
    missing_after = df.isnull().sum()
    total_missing_after = missing_after.sum()
    
    print(f"Missing values before: {total_missing_before} | After imputation: {total_missing_after}")
    if return_metrics:
        return df, {"before": missing_before.to_dict(), "after": missing_after.to_dict()}
    return df

def handle_outliers(df: pd.DataFrame, return_metrics=False):
    print("\n--- OUTLIER HANDLING ---")
    cols_to_clip = ['Amount', 'Time']
    
    df_before_outliers_amount = df['Amount'].copy()
    df_before_outliers_time = df['Time'].copy()
    
    outliers_clipped_dict = {}
    
    for col in cols_to_clip:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"Outliers detected in {col} (IQR method): {outliers_count}")
        
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        
        clipped_count = ((df[col] < p1) | (df[col] > p99)).sum()
        outliers_clipped_dict[col] = clipped_count
        
        df[col] = df[col].clip(lower=p1, upper=p99)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.boxplot(y=df_before_outliers_amount, ax=axes[0, 0], color='lightblue').set_title('Amount (Before)')
    sns.boxplot(y=df['Amount'], ax=axes[0, 1], color='lightgreen').set_title('Amount (After Winsorization)')
    sns.boxplot(y=df_before_outliers_time, ax=axes[1, 0], color='lightblue').set_title('Time (Before)')
    sns.boxplot(y=df['Time'], ax=axes[1, 1], color='lightgreen').set_title('Time (After Winsorization)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "outliers_before_after.png"))
    plt.close()
    
    if return_metrics:
        return df, outliers_clipped_dict
    return df

def remove_duplicates(df: pd.DataFrame, return_metrics=False):
    print("\n--- NOISE HANDLING ---")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicate_count}")
    
    shape_before = df.shape
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
    shape_after = df.shape
    
    print(f"Shape before dropping duplicates: {shape_before}")
    print(f"Shape after dropping duplicates: {shape_after}")
    if return_metrics:
        return df, duplicate_count
    return df

def full_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 50)
    print("DATA PREPROCESSING")
    print("-" * 50)
    
    df, missing_metrics = check_missing(df, return_metrics=True)
    df, outliers_clipped_dict = handle_outliers(df, return_metrics=True)
    df, duplicate_count = remove_duplicates(df, return_metrics=True)
    
    print("\n--- PREPROCESSING SUMMARY TABLE ---")
    summary_data = []
    
    cols_to_clip = ['Amount', 'Time']
    for i, col in enumerate(df.columns):
        mb = missing_metrics["before"].get(col, 0)
        ma = missing_metrics["after"].get(col, 0)
        out_clipped = outliers_clipped_dict.get(col, "N/A") if col in cols_to_clip else "N/A"
        dups = duplicate_count if i == 0 else "-"
        summary_data.append([col, mb, ma, out_clipped, dups])
        
    summary_df = pd.DataFrame(summary_data, columns=['column', 'missing_before', 'missing_after', 'outliers_clipped', 'duplicates_removed'])
    print(summary_df.to_string(index=False))

    return df
