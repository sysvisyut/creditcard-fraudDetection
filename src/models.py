from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def train_with_class_weights(X_train, y_train):
    # Placeholder for future requirements
    pass

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
