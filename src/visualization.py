import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
import config

def plot_class_distribution(y):
    print("-" * 50)
    print("VISUALIZING CLASS DISTRIBUTION")
    print("-" * 50)
    
    os.makedirs(config.OUTPUT_PLOTS, exist_ok=True)
    
    class_counts = y.value_counts()
    count_0 = class_counts.get(0, 0)
    count_1 = class_counts.get(1, 0)
    
    labels = ['Legitimate (0)', 'Fraud (1)']
    counts = [count_0, count_1]
    
    colors = ['#2ca02c', '#d62728'] 
    
    # 1. Bar chart
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, hue=labels, palette=colors, legend=False)
    plt.title('Count of Legitimate vs Fraudulent Transactions')
    plt.xlabel('Transaction Class')
    plt.ylabel('Count')
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')
                    
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(colors))]
    plt.legend(handles, labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "1_bar_chart.png"))
    plt.close()
    
    # 2. Pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.2f%%', colors=colors, startangle=140)
    plt.title('Percentage Share of Fraud vs Legitimate Transactions')
    plt.legend(labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "2_pie_chart.png"))
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
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "3_donut_chart.png"))
    plt.close()
    
    # 4. Log-scale bar chart
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, hue=labels, palette=colors, legend=False)
    plt.yscale('log')
    plt.title('Log Scale: Legitimate vs Fraudulent Transactions')
    plt.xlabel('Transaction Class')
    plt.ylabel('Count (Log Scale)')
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), 
                    textcoords='offset points')
                    
    plt.legend(handles, labels, title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "4_log_scale_bar_chart.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred_dict, title="Baseline Confusion Matrices"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "baseline_confusion_matrices.png"))
    plt.close()

def plot_roc_curve(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('Baseline Models ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "baseline_roc_curves.png"))
    plt.close()

def plot_pr_curve(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.title('Baseline Models Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "baseline_pr_curves.png"))
    plt.close()

def plot_threshold_curves(threshold_df, default_threshold=0.5, optimal_threshold=None):
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', color='blue', lw=2)
    plt.plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall', color='green', lw=2)
    plt.plot(threshold_df['Threshold'], threshold_df['F1'], label='F1-Score', color='red', lw=2)
    
    plt.axvline(x=default_threshold, color='black', linestyle='--', label=f'Default ({default_threshold})', alpha=0.7)
    if optimal_threshold is not None:
        plt.axvline(x=optimal_threshold, color='purple', linestyle=':', label=f'Optimal ({optimal_threshold})', alpha=0.9, lw=2)
        
    plt.title('Performance Metrics vs Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "threshold_tuning_curves.png"))
    plt.close()

def plot_threshold_confusion_matrices(y_true, y_prob, optimal_threshold, default_threshold=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    y_pred_default = (y_prob >= default_threshold).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)
    sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title(f'Default Threshold ({default_threshold})')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title(f'Optimal Threshold ({optimal_threshold})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "threshold_confusion_matrices.png"))
    plt.close()

def plot_feature_importance(model, features):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = min(20, len(features))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [features[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "feature_importance.png"))
    plt.close()

def plot_sampling_confusion_matrices(y_true, y_pred_dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "sampling_confusion_matrices.png"))
    plt.close()

def plot_sampling_roc_curves(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('Sampling Models ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "sampling_roc_curves.png"))
    plt.close()

def plot_sampling_pr_curves(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.title('Sampling Models Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "sampling_pr_curves.png"))
    plt.close()

def plot_classweight_confusion_matrices(y_true, y_pred_dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "classweight_confusion_matrices.png"))
    plt.close()

def plot_classweight_roc_curves(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('Cost-Sensitive Models ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "classweight_roc_curves.png"))
    plt.close()

def plot_classweight_pr_curves(models_prob_dict, X, y):
    plt.figure(figsize=(10, 8))
    for name, y_prob in models_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y, y_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.title('Cost-Sensitive Models Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "classweight_pr_curves.png"))
    plt.close()

def plot_final_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Final Model Confusion Matrix (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "final_confusion_matrix.png"))
    plt.close()

def plot_final_roc_curve(y_true, y_prob):
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, lw=2, color='darkorange', label=f"Final Model (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.title('ROC Curve (Test Set)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "final_roc_curve.png"))
    plt.close()

def plot_final_pr_curve(y_true, y_prob):
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, lw=2, color='green', label=f"Final Model (PR-AUC = {pr_auc:.4f})")
    plt.title('Precision-Recall Curve (Test Set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "final_pr_curve.png"))
    plt.close()

def plot_roc_vs_pr_comparison(y_true, y_prob):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color='darkorange', label=f"AUC = {roc_auc_val:.4f}")
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall, precision)
    axes[1].plot(recall, precision, lw=2, color='green', label=f"PR-AUC = {pr_auc_val:.4f}")
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "roc_vs_pr_comparison.png"))
    plt.close()

def plot_final_feature_importance(model, features):
    # Depending on model architecture (XGB vs RF), attributes differ mathematically:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = min(15, len(features))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [features[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_importances, y=top_features, hue=top_features, palette="viridis", legend=False)
    plt.title(f'Top {top_n} Feature Importances - Final Model')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "final_feature_importance.png"))
    plt.close()

def plot_clustering_analysis(X_test, y_pred_cluster, y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    
    plt.figure(figsize=(14, 6))
    
    # Using Subplots correctly
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_true, palette=['blue', 'red'], alpha=0.6, s=20)
    plt.title('True Labels (PCA 2D)')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred_cluster, palette=['blue', 'purple'], alpha=0.6, s=20)
    plt.title('KMeans Cluster Assignments (PCA 2D)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "clustering_analysis.png"))
    plt.close()

def plot_master_comparison(master_df):
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(master_df['Model']))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, master_df['F1'], width, label='F1-Score', color='teal')
    rects2 = ax.bar(x + width/2, master_df['ROC-AUC'], width, label='ROC-AUC', color='coral')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: F1 vs ROC-AUC across all techniques')
    ax.set_xticks(x)
    ax.set_xticklabels(master_df['Model'], rotation=45, ha="right")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "master_model_comparison.png"))
    plt.close()

