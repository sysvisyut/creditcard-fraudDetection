import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
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
    ax = sns.barplot(x=labels, y=counts, palette=colors)
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
    ax = sns.barplot(x=labels, y=counts, palette=colors)
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

def plot_threshold_curves(thresholds, metrics):
    pass

def plot_feature_importance(model, features):
    pass

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
