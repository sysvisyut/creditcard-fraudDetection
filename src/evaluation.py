import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
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

def compute_clustering_score(X_test):
    # Placeholder
    pass

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
