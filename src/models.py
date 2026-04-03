from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import os
import config

def train_baseline_models(X_train, y_train):
    print("\nTraining Baseline Logistic Regression...")
    lr = LogisticRegression(max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    
    print("Training Baseline Random Forest...")
    rf = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    return {'Logistic Regression': lr, 'Random Forest': rf}

def train_sampling_models(sampling_variants):
    print("\n" + "-" * 50)
    print("TRAINING SAMPLING VARIANTS (RANDOM FOREST)")
    print("-" * 50)
    
    sampling_models = {}
    
    for variant in ['smote', 'random_oversample', 'random_undersample']:
        print(f"Training Random Forest on {variant} data...")
        X_train, y_train = sampling_variants[variant]
        rf = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        sampling_models[variant] = rf
        
    return sampling_models

def train_with_class_weights(X_train, y_train):
    print("\n" + "-" * 50)
    print("TRAINING COST-SENSITIVE MODELS (CLASS WEIGHTS)")
    print("-" * 50)
    
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    scale_pos_weight = count_0 / count_1
    
    print(f"Calculated scale_pos_weight (Class 0 / Class 1): {scale_pos_weight:.4f}")
    
    print("Training Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight='balanced', max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    
    print("Training Random Forest (class_weight='balanced')...")
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Training XGBoost (scale_pos_weight)...")
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    
    return {
        'Logistic Regression (CW)': lr, 
        'Random Forest (CW)': rf,
        'XGBoost (CW)': xgb_model
    }
def train_final_model(X_train, y_train):
    # Placeholder for future requirements
    pass

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
