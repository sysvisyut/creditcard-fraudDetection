import os
import warnings
import pickle
import pandas as pd
import numpy as pd_numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
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

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Data Transformation and Integration.
    """
    print("-" * 50)
    print("DATA TRANSFORMATION AND INTEGRATION")
    print("-" * 50)
    
    # --- SCALING ---
    # WHY we scale: V1-V28 are already PCA-scaled components, meaning they are centered and scaled.
    # The 'Amount' and 'Time' columns are in their original units, which vary widely in scale and magnitude.
    # Scaling Amount and Time ensures ALL our features are on a comparable scale, 
    # preventing features with larger magnitudes from dominating the learning process
    # and ensuring distance-based or gradient descent algorithms perform optimally.
    
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['Amount_Scaled'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler_time.fit_transform(df[['Time']])
    
    # Drop original 'Amount' and 'Time' columns
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    # --- DATA INTEGRATION ---
    print("\n--- DATA INTEGRATION ---")
    print("[1] All features are confirmed to be in a single unified dataframe.")
    print(f"\n[2] Final dataset shape: {df.shape}")
    print("\n[3] Final column list:")
    print(", ".join(df.columns.tolist()))
    
    print("\n[4] Final dtypes:")
    print(df.dtypes)
    
    # --- VERIFICATION ---
    print("\n--- VERIFICATION ---")
    
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # Plot distribution of Amount_Scaled and Time_Scaled (histograms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['Amount_Scaled'], bins=50, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Distribution of Amount_Scaled')
    
    sns.histplot(df['Time_Scaled'], bins=50, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Distribution of Time_Scaled')
    
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "scaled_distributions.png"))
    plt.close()
    
    # Print min, max, mean, std of Amount_Scaled and Time_Scaled
    print("\nScaled Feature Statistics:")
    print(f"{'Feature':<15} | {'Min':>10} | {'Max':>10} | {'Mean':>10} | {'Std':>10}")
    print("-" * 65)
    for col in ['Amount_Scaled', 'Time_Scaled']:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        col_std = df[col].std()
        print(f"{col:<15} | {col_min:>10.4f} | {col_max:>10.4f} | {col_mean:>10.4f} | {col_std:>10.4f}")
    
    print("\nData Transformation complete. All features are now on comparable scales. Dataset ready for feature selection.\n")
    
    return df
def select_features(df: pd.DataFrame):
    """
    Perform Feature Selection and Dimensionality Reduction.
    """
    print("-" * 50)
    print("FEATURE SELECTION & DIMENSIONALITY REDUCTION")
    print("-" * 50)
    
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # --- CORRELATION ANALYSIS ---
    print("\n--- CORRELATION ANALYSIS ---")
    corr_matrix = df.corr()
    corr_with_class = corr_matrix['Class'].drop('Class')
    
    # Sort by absolute value for printing and plotting
    corr_sorted_abs = corr_with_class.abs().sort_values(ascending=True) # Ascending for horizontal bar
    corr_sorted_desc = corr_with_class.reindex(corr_with_class.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 8))
    corr_with_class.loc[corr_sorted_abs.index].plot(kind='barh', color='skyblue')
    plt.title('Feature Correlation with Target (Class) - Sorted by Absolute Value')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "feature_correlation.png"))
    plt.close()
    
    print("Top 10 Most Correlated Features with Fraud (Class):")
    for feat, val in corr_sorted_desc.head(10).items():
        print(f"  {feat}: {val:.4f}")
        
    # --- FEATURE IMPORTANCE (Tree-based) ---
    print("\n--- FEATURE IMPORTANCE (Tree-based) ---")
    X_full = df.drop('Class', axis=1)
    y_full = df['Class']
    
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_full, y_full)
    
    importances = pd.Series(rf.feature_importances_, index=X_full.columns)
    importances_sorted = importances.sort_values(ascending=False)
    
    top_20_features = importances_sorted.head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_20_features.values, y=top_20_features.index, palette='viridis')
    plt.title('Top 20 Feature Importances (RandomForest)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "feature_importance.png"))
    plt.close()
    
    # Select top 20 features
    selected_features = top_20_features.index.tolist()
    
    # --- DIMENSIONALITY REDUCTION (PCA for visualization only) ---
    print("\n--- DIMENSIONALITY REDUCTION (PCA Visualization) ---")
    X_selected = X_full[selected_features]
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_selected)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    df_pca['Class'] = y_full.values
    
    # Sample 5000 points for speed
    if len(df_pca) > 5000:
        df_pca_sample = df_pca.sample(n=5000, random_state=42)
    else:
        df_pca_sample = df_pca
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Class', 
        palette={0: 'blue', 1: 'red'},
        data=df_pca_sample, 
        alpha=0.6,
        edgecolor=None
    )
    plt.title('2D PCA Visualization of Selected Features (Sampled 5000 points)')
    
    # Custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
    plt.legend(handles, ['Legitimate (0)', 'Fraud (1)'], title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "pca_2d_visualization.png"))
    plt.close()
    
    print("PCA Explained Variance Ratio:")
    print(f"  Component 1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  Component 2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  Total Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # --- FINAL FEATURE SET ---
    print("\n--- FINAL FEATURE SET ---")
    print("Final list of selected features being used for modeling:")
    print(", ".join(selected_features))
    
    X_final = X_full[selected_features]
    y_final = y_full
    
    print(f"\nShape of X (features): {X_final.shape}")
    print(f"Shape of y (target): {y_final.shape}")
    
    return X_final, y_final
def prepare_modeling_data(X: pd.DataFrame, y: pd.Series):
    """
    Perform Train/Val/Test Splitting, Bias-Variance Analysis, and Sampling.
    """
    print("-" * 50)
    print("DATA SPLITTING & SAMPLING STRATEGIES")
    print("-" * 50)
    
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # --- 1. TRAIN/VAL/TEST SPLIT ---
    print("\n--- 1. TRAIN/VAL/TEST SPLIT ---")
    
    # First split: Train 70%, Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    # Second split: Val 15%, Test 15% (50% of the Temp 30%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    def print_class_dist(name, current_y):
        dist = current_y.value_counts(normalize=True) * 100
        print(f"  {name} Class Dist -> 0: {dist.get(0, 0):.2f}%, 1: {dist.get(1, 0):.2f}%")
        
    print("\nClass Distributions (Stratification Check):")
    print_class_dist("Train", y_train)
    print_class_dist("Val", y_val)
    print_class_dist("Test", y_test)
    
    # --- 2. BIAS-VARIANCE TRADEOFF ANALYSIS ---
    print("\n--- 2. BIAS-VARIANCE TRADEOFF ANALYSIS ---")
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    train_f1_scores = []
    val_f1_scores = []
    
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("Training LogisticRegression across varying C values...")
    best_c = C_values[0]
    best_val_f1 = -1
    
    for c in C_values:
        lr = LogisticRegression(C=c, max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train, y_train)
        
        train_preds = lr.predict(X_train)
        val_preds = lr.predict(X_val)
        
        t_f1 = f1_score(y_train, train_preds)
        v_f1 = f1_score(y_val, val_preds)
        
        train_f1_scores.append(t_f1)
        val_f1_scores.append(v_f1)
        
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_c = c
            
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_f1_scores, marker='o', label='Train F1 Score')
    plt.plot(C_values, val_f1_scores, marker='s', label='Validation F1 Score')
    plt.xscale('log')
    plt.title('Bias-Variance Tradeoff Analysis (Logistic Regression)')
    plt.xlabel('C value (Inverse of Regularization Strength)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "plots", "bias_variance_tradeoff.png"))
    plt.close()
    
    print(f"\nOptimal C value (highest Validation F1): {best_c}")
    print("Interpretation: Low C = high bias (underfitting), High C = high variance (overfitting)")
    
    # --- 3. SAMPLING STRATEGIES ---
    print("\n--- 3. SAMPLING STRATEGIES ---")
    sampling_variants = {}
    
    print(f"Original Training Class Distribution: {dict(y_train.value_counts())}")
    sampling_variants['original'] = (X_train, y_train)
    
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE Class Distribution: {dict(y_train_sm.value_counts())}")
    sampling_variants['smote'] = (X_train_sm, y_train_sm)
    
    print("\nApplying RandomOverSampler...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    print(f"After RandomOverSampler Class Distribution: {dict(y_train_ros.value_counts())}")
    sampling_variants['random_oversample'] = (X_train_ros, y_train_ros)
    
    print("\nApplying RandomUnderSampler...")
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    print(f"After RandomUnderSampler Class Distribution: {dict(y_train_rus.value_counts())}")
    sampling_variants['random_undersample'] = (X_train_rus, y_train_rus)

    return sampling_variants, X_val, y_val, X_test, y_test

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
    
    # Data Transformation and Integration
    df = transform_data(df)
    
    # Feature Selection and Dimensionality Reduction
    X, y = select_features(df)
    
    # Data Splitting, Tradeoff Analysis, and Sampling
    sampling_variants, X_val, y_val, X_test, y_test = prepare_modeling_data(X, y)
if __name__ == "__main__":
    main()
