import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import config

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 50)
    print("DATA TRANSFORMATION AND INTEGRATION")
    print("-" * 50)
    
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    
    df['Amount_Scaled'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time_Scaled'] = scaler_time.fit_transform(df[['Time']])
    
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    
    print("\n--- DATA INTEGRATION ---")
    print("[1] All features are confirmed to be in a single unified dataframe.")
    print(f"\n[2] Final dataset shape: {df.shape}")
    print("\n[3] Final column list:")
    print(", ".join(df.columns.tolist()))
    
    print("\n[4] Final dtypes:")
    print(df.dtypes)
    
    print("\n--- VERIFICATION ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['Amount_Scaled'], bins=50, kde=True, ax=axes[0], color='purple')
    axes[0].set_title('Distribution of Amount_Scaled')
    
    sns.histplot(df['Time_Scaled'], bins=50, kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Distribution of Time_Scaled')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "scaled_distributions.png"))
    plt.close()
    
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

def select_features(X_full: pd.DataFrame, y_full: pd.Series) -> list:
    print("-" * 50)
    print("FEATURE SELECTION & DIMENSIONALITY REDUCTION")
    print("-" * 50)
    
    print("\n--- CORRELATION ANALYSIS ---")
    df = X_full.copy()
    df['Class'] = y_full
    corr_matrix = df.corr()
    corr_with_class = corr_matrix['Class'].drop('Class')
    
    corr_sorted_abs = corr_with_class.abs().sort_values(ascending=True)
    corr_sorted_desc = corr_with_class.reindex(corr_with_class.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 8))
    corr_with_class.loc[corr_sorted_abs.index].plot(kind='barh', color='skyblue')
    plt.title('Feature Correlation with Target (Class) - Sorted by Absolute Value')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "feature_correlation.png"))
    plt.close()
    
    print("Top 10 Most Correlated Features with Fraud (Class):")
    for feat, val in corr_sorted_desc.head(10).items():
        print(f"  {feat}: {val:.4f}")
        
    print("\n--- FEATURE IMPORTANCE (Tree-based) ---")
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=config.RF_MAX_DEPTH, random_state=config.RANDOM_STATE, n_jobs=-1)
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
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "feature_importance.png"))
    plt.close()
    
    selected_features = top_20_features.index.tolist()
    
    print("\n--- FINAL FEATURE SET ---")
    print("Final list of selected features being used for modeling:")
    print(", ".join(selected_features))
    
    return selected_features

def reduce_dimensions(X: pd.DataFrame, y: pd.Series):
    print("\n--- DIMENSIONALITY REDUCTION (PCA Visualization) ---")
    
    pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    df_pca['Class'] = y.values
    
    if len(df_pca) > config.SAMPLING_N_SAMPLES_PCA:
        df_pca_sample = df_pca.sample(n=config.SAMPLING_N_SAMPLES_PCA, random_state=config.RANDOM_STATE)
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
    plt.title(f'2D PCA Visualization of Selected Features (Sampled {config.SAMPLING_N_SAMPLES_PCA} points)')
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
    plt.legend(handles, ['Legitimate (0)', 'Fraud (1)'], title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "pca_2d_visualization.png"))
    plt.close()
    
    print("PCA Explained Variance Ratio:")
    print(f"  Component 1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  Component 2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"  Total Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")
