import pytest
from src.preprocessing import full_preprocessing_pipeline
from src.features import scale_features, select_features
from src.sampling import split_data, apply_smote
from src.models import train_baseline_models, save_model, load_model
from src.evaluation import evaluate_model, build_master_comparison_table, tune_threshold
import config
import os

def test_full_pipeline_runs_without_error(raw_df):
    clean = full_preprocessing_pipeline(raw_df.copy())
    scaled = scale_features(clean)
    X = scaled.drop('Class', axis=1)
    y = scaled['Class']
    features = select_features(X, y)
    X_reduced = X[features]
    res = split_data(X_reduced, y)
    assert res[0].shape[0] > 0

def test_preprocessing_to_features_chain(raw_df):
    clean = full_preprocessing_pipeline(raw_df.copy())
    scaled = scale_features(clean)
    assert 'Amount_Scaled' in scaled.columns

def test_features_to_sampling_chain(clean_df):
    scaled = scale_features(clean_df.copy())
    X = scaled.drop('Class', axis=1)
    y = scaled['Class']
    features = select_features(X, y)
    X_reduced = X[features]
    res = split_data(X_reduced, y)
    assert len(res) == 6
    assert str(res[0].dtypes.iloc[0]) in ['float64', 'int64']

def test_sampling_to_model_chain(X_train, y_train):
    X_sm, y_sm = apply_smote(X_train, y_train)
    models = train_baseline_models(X_sm, y_sm)
    assert isinstance(models, dict)

def test_model_to_evaluation_chain(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    keys = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    for k in keys:
        assert k in res

def test_evaluation_to_visualization_chain():
    mock_res = [{'Model': 'Mock', 'Accuracy': 1, 'Precision': 1, 'Recall': 1, 'F1': 1, 'ROC-AUC': 1}]
    df = build_master_comparison_table(mock_res)
    assert df.shape == (1, 6)

def test_threshold_tuning_uses_val_not_test(trained_rf_model, X_val, y_val, X_test, y_test, monkeypatch):
    called = []
    original_predict = trained_rf_model.predict_proba
    def mock_predict(X):
        called.append(X.shape[0])
        return original_predict(X)
    monkeypatch.setattr(trained_rf_model, "predict_proba", mock_predict)
    tune_threshold(trained_rf_model, X_val, y_val)
    assert called[0] == X_val.shape[0]

def test_saved_model_produces_same_predictions(trained_baseline_model, X_test):
    path = os.path.join(config.OUTPUT_MODELS, "integration_test_model.pkl")
    save_model(trained_baseline_model, path)
    loaded = load_model(path)
    
    y_pred_original = trained_baseline_model.predict(X_test)
    y_pred_loaded = loaded.predict(X_test)
    
    assert (y_pred_original == y_pred_loaded).all()
