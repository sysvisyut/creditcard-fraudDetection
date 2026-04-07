import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning
import config

def print_class_dist(name, current_y):
    dist = current_y.value_counts(normalize=True) * 100
    print(f"  {name} Class Dist -> 0: {dist.get(0, 0):.2f}%, 1: {dist.get(1, 0):.2f}%")

def split_data(X: pd.DataFrame, y: pd.Series):
    print("-" * 50)
    print("DATA SPLITTING")
    print("-" * 50)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config.TRAIN_TEST_SPLIT_TEMP, stratify=y, random_state=config.RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config.TRAIN_TEST_SPLIT_VAL, stratify=y_temp, random_state=config.RANDOM_STATE)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    print("\nClass Distributions (Stratification Check):")
    print_class_dist("Train", y_train)
    print_class_dist("Val", y_val)
    print_class_dist("Test", y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    print("\nApplying SMOTE...") #synthetic minority oversampling technique
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE Class Distribution: {dict(y_train_sm.value_counts())}")
    return X_train_sm, y_train_sm

def apply_oversample(X_train: pd.DataFrame, y_train: pd.Series):
    print("\nApplying RandomOverSampler...")
    ros = RandomOverSampler(random_state=config.RANDOM_STATE)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    print(f"After RandomOverSampler Class Distribution: {dict(y_train_ros.value_counts())}")
    return X_train_ros, y_train_ros

def apply_undersample(X_train: pd.DataFrame, y_train: pd.Series):
    print("\nApplying RandomUnderSampler...")
    rus = RandomUnderSampler(random_state=config.RANDOM_STATE)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    print(f"After RandomUnderSampler Class Distribution: {dict(y_train_rus.value_counts())}")
    return X_train_rus, y_train_rus

def bias_variance_analysis(X_train, y_train, X_val, y_val):
    print("\n--- BIAS-VARIANCE TRADEOFF ANALYSIS ---")
    
    train_f1_scores = []
    val_f1_scores = []
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    print("Training LogisticRegression across varying C values...")
    best_c = config.LR_C_VALUES[0]
    best_val_f1 = -1
    
    for c in config.LR_C_VALUES:
        lr = LogisticRegression(C=c, max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_STATE, class_weight='balanced')
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
    plt.plot(config.LR_C_VALUES, train_f1_scores, marker='o', label='Train F1 Score')
    plt.plot(config.LR_C_VALUES, val_f1_scores, marker='s', label='Validation F1 Score')
    plt.xscale('log')
    plt.title('Bias-Variance Tradeoff Analysis (Logistic Regression)')
    plt.xlabel('C value (Inverse of Regularization Strength)')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_PLOTS, "bias_variance_tradeoff.png"))
    plt.close()
    
    print(f"\nOptimal C value (highest Validation F1): {best_c}")
    print("Interpretation: Low C = high bias (underfitting), High C = high variance (overfitting)")
