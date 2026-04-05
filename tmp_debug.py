import pandas as pd
import config
from src.data_loader import load_data, identify_features
from src.preprocessing import full_preprocessing_pipeline
from src.features import scale_features, select_features
from src.sampling import split_data, apply_smote
from src.models import train_final_model, load_model
from src.evaluation import evaluate_final_model, evaluate_model

# 1. Load Data
df = load_data(config.DATA_PATH)
predictor_cols, target_col = identify_features(df)
df = full_preprocessing_pipeline(df)
df = scale_features(df)
selected_features = select_features(df.drop(target_col, axis=1), df[target_col])
X_final = df[selected_features]
y_final = df[target_col]
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_final, y_final)

try:
    final_model = load_model(config.OUTPUT_MODELS + "final_fraud_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Cannot load model", e)

try:
    final_eval_results = evaluate_final_model(final_model, X_test, y_test, threshold=0.25)
    print("Final eval results:", final_eval_results)
except Exception as e:
    print("Error evaluating final model:", e)
