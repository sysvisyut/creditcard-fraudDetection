import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score, matthews_corrcoef, silhouette_score
from sklearn.cluster import KMeans
import os
import config

def evaluate_model(model, model_name, X, y, threshold=0.5) -> dict:
    
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label=1)
    rec = recall_score(y, y_pred, pos_label=1)
    f1 = f1_score(y, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y, y_prob)
    
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y, y_pred))
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

import numpy as np

def tune_threshold(model, X_val, y_val):
    print("\n" + "-" * 50)
    print("THRESHOLD ANALYSIS")
    print("-" * 50)
    
    y_prob = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        # Avoid division by zero warnings if precision collapses
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        
        results.append({
            'Threshold': round(t, 2),
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
        
    threshold_df = pd.DataFrame(results)
    
    optimal_idx = threshold_df['F1'].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']
    
    # Maximize recall ideally, but let's just pick the max mathematically
    recall_idx = threshold_df['Recall'].idxmax()
    high_recall_threshold = threshold_df.loc[recall_idx, 'Threshold']
    
    print(f"Optimal Threshold (Max F1): {optimal_threshold}")
    print(f"High Recall Threshold (Max Recall): {high_recall_threshold}")
    
    return threshold_df, optimal_threshold, high_recall_threshold

def evaluate_final_model(model, X_test, y_test, threshold):
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 50)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"MCC:       {mcc:.4f}")
    
    print("\nFINAL Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def compute_clustering_score(X_test, y_test):
    print("\n" + "-" * 50)
    print("CLUSTERING SCORE ANALYSIS")
    print("-" * 50)
    
    # We sample if dataset is too large, but 15% Test set ~42K rows is fine for K-Means
    kmeans = KMeans(n_clusters=2, random_state=config.RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_test)
    
    # Silhouette score can be extremely slow on 42k rows, sampling to 10k 
    sample_size = min(10000, X_test.shape[0])
    np.random.seed(config.RANDOM_STATE)
    idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
    
    X_sample = X_test.iloc[idx] if isinstance(X_test, pd.DataFrame) else X_test[idx]
    labels_sample = cluster_labels[idx]
    
    sil_score = silhouette_score(X_sample, labels_sample)
    
    print(f"Silhouette Score: {sil_score:.2f} — measures how well the model separates fraud vs legit")
    
    return cluster_labels

def build_master_comparison_table(all_results: list) -> pd.DataFrame:
    clean_results = []
    for r in all_results:
        clean_results.append({
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1': r['F1'],
            'ROC-AUC': r['ROC-AUC']
        })
    results_df = pd.DataFrame(clean_results)
    
    print("\n--- BASELINE RESULTS SUMMARY ---")
    print(results_df.to_string(index=False))
    
    print("\nBaseline without balancing — high accuracy but poor recall on fraud class due to class imbalance.")
    return results_df
