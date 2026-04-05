import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, average_precision_score, matthews_corrcoef, silhouette_score
from sklearn.cluster import KMeans
import os
import config

def evaluate_model(model, model_name, X, y, threshold=0.5) -> dict:
    try:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, pos_label=1)
        rec = recall_score(y, y_pred, pos_label=1)
        f1 = f1_score(y, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y, y_prob)
        
        # Ensure we print the explicit classification report particularly checking RUS or others
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y, y_pred))
        
        if 'random_undersample' in model_name.lower():
            print(f"[DEBUG] Raw predictions type for {model_name}: {y_pred.dtype}")
            print(f"[DEBUG] Unique predict labels: {np.unique(y_pred)}")
            
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
    except Exception as e:
        import traceback
        print("\n" + "!" * 50)
        print(f"CRITICAL ERROR evaluating {model_name}: {e}")
        traceback.print_exc()
        print("!" * 50 + "\n")
        
        return {
            'Model': model_name,
            'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0, 'ROC-AUC': 0.0,
            'y_pred': np.zeros_like(y), 'y_prob': np.zeros_like(y, dtype=float)
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
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'ROC-AUC': roc_auc,
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
    
    return sil_score, cluster_labels

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

def generate_master_comparison(all_results, sampling_results, cw_results, final_eval_results, optimal_threshold):
    print("\n" + "=" * 50)
    print("MASTER MODEL COMPARISON TABLE")
    print("=" * 50)
    
    rows = []
    
    # 1. Baseline - Logistic Regression
    lr_base = next(r for r in all_results if r['Model'] == 'Logistic Regression')
    rows.append({
        'Model': 'Baseline - Logistic Regression', 'Sampling': 'None', 'Class Weight': 'None', 'Threshold': 0.5,
        'Accuracy': lr_base['Accuracy'], 'Precision': lr_base['Precision'], 'Recall': lr_base['Recall'], 
        'F1': lr_base['F1'], 'ROC-AUC': lr_base['ROC-AUC']
    })
    
    # 2. Baseline - Random Forest
    rf_base = next(r for r in all_results if r['Model'] == 'Random Forest')
    rows.append({
        'Model': 'Baseline - Random Forest', 'Sampling': 'None', 'Class Weight': 'None', 'Threshold': 0.5,
        'Accuracy': rf_base['Accuracy'], 'Precision': rf_base['Precision'], 'Recall': rf_base['Recall'], 
        'F1': rf_base['F1'], 'ROC-AUC': rf_base['ROC-AUC']
    })
    
    # 3-5. Sampling (RF)
    for r in sampling_results:
        sampling_method = 'SMOTE' if r['Model'] == 'smote' else ('RandomOverSampler' if r['Model'] == 'random_oversample' else 'RandomUnderSampler')
        rows.append({
            'Model': f"RF + {sampling_method}", 'Sampling': sampling_method, 'Class Weight': 'None', 'Threshold': 0.5,
            'Accuracy': r['Accuracy'], 'Precision': r['Precision'], 'Recall': r['Recall'], 
            'F1': r['F1'], 'ROC-AUC': r['ROC-AUC']
        })
        
    # 6-8. Class Weights
    for r in cw_results:
        name_map = {
            'Logistic Regression (CW)': 'LR + class_weight=balanced',
            'Random Forest (CW)': 'RF + class_weight=balanced',
            'XGBoost (CW)': 'XGBoost + scale_pos_weight'
        }
        cw_val = 'scale_pos_weight' if 'XGBoost' in r['Model'] else 'balanced'
        rows.append({
            'Model': name_map[r['Model']], 'Sampling': 'None', 'Class Weight': cw_val, 'Threshold': 0.5,
            'Accuracy': r['Accuracy'], 'Precision': r['Precision'], 'Recall': r['Recall'], 
            'F1': r['F1'], 'ROC-AUC': r['ROC-AUC']
        })
        
    # 9. Final Model
    rows.append({
        'Model': 'Best Model + optimal threshold (FINAL)', 'Sampling': 'None', 'Class Weight': 'scale_pos_weight', 'Threshold': optimal_threshold,
        'Accuracy': final_eval_results.get('Accuracy', 0), 'Precision': final_eval_results.get('Precision', 0), 
        'Recall': final_eval_results.get('Recall', 0), 'F1': final_eval_results.get('F1', 0), 
        'ROC-AUC': final_eval_results.get('ROC-AUC', 0)
    })
    
    master_df = pd.DataFrame(rows)
    print(master_df.to_string(index=False))
    
    print("\nBest Scores per Metric:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']:
        best_row = master_df.loc[master_df[metric].idxmax()]
        print(f"Best {metric}: {best_row[metric]:.4f} ({best_row['Model']})")
        
    return master_df

def generate_business_impact_report(y_test, y_pred, y_prob, optimal_threshold, final_model_name):
    print("\n\n" + "=" * 50)
    print("=== BUSINESS IMPACT REPORT ===")
    print("=" * 50)
    
    total_transactions = len(y_test)
    fraudulent = sum(y_test)
    fraud_pct = (fraudulent / total_transactions) * 100
    
    print(f"1. Dataset: {total_transactions} total transactions, {fraudulent} fraudulent ({fraud_pct:.2f}%)")
    
    recall = recall_score(y_test, y_pred)
    print(f"2. Best Model Recall: {recall*100:.2f}% — meaning the model catches {recall*100:.2f}% of all fraud cases")
    
    fn = sum((y_test == 1) & (y_pred == 0))
    print(f"3. False Negatives (missed fraud): {fn} transactions out of test set")
    
    fp = sum((y_test == 0) & (y_pred == 1))
    print(f"4. False Positives (false alarms): {fp} transactions out of test set")
    
    average_fraud_amount = 122
    fraud_caught = sum((y_test == 1) & (y_pred == 1))
    potential_caught = fraud_caught * average_fraud_amount
    cost_false_alarms = fp * 10
    net_benefit = potential_caught - cost_false_alarms
    
    print(f"5. Assuming average fraud amount = ${average_fraud_amount} (dataset mean):")
    print(f"   - Potential fraud caught = {fraud_caught}_caught × ${average_fraud_amount} = ${potential_caught}")
    print(f"   - Cost of false alarms (investigation cost ~10 each) = {fp} × $10 = ${cost_false_alarms}")
    print(f"   - Net benefit of deploying this model = ${net_benefit}")
    
    print(f"\n6. Recommendation: \"Deploy the {final_model_name} with threshold={optimal_threshold} for optimal")
    print("   fraud detection. Monitor recall monthly and retrain quarterly.\"")
    print("=" * 50)
