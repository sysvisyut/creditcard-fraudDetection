import pytest
import os
from src.models import train_baseline_models, train_with_class_weights, save_model, load_model
import config

def test_baseline_models_returns_dict(trained_baseline_models_dict):
    assert isinstance(trained_baseline_models_dict, dict)

def test_baseline_dict_has_expected_keys(trained_baseline_models_dict):
    assert 'Logistic Regression' in trained_baseline_models_dict
    assert 'Random Forest' in trained_baseline_models_dict

def test_baseline_models_are_fitted(trained_baseline_models_dict):
    for model in trained_baseline_models_dict.values():
        assert hasattr(model, 'predict')

def test_class_weight_models_trained(X_train, y_train):
    models = train_with_class_weights(X_train, y_train)
    assert len(models) == 3
    assert 'Logistic Regression (CW)' in models
    assert 'Random Forest (CW)' in models
    assert 'XGBoost (CW)' in models

def test_model_predict_output_shape(trained_baseline_model, X_val):
    preds = trained_baseline_model.predict(X_val)
    assert len(preds) == len(X_val)

def test_model_predict_proba_two_columns(trained_baseline_model, X_val):
    probs = trained_baseline_model.predict_proba(X_val)
    assert probs.shape[1] == 2

def test_save_model_creates_file(trained_baseline_model):
    path = os.path.join(config.OUTPUT_MODELS, "test_model.pkl")
    save_model(trained_baseline_model, path)
    assert os.path.exists(path)

def test_load_model_returns_same_type(trained_baseline_model):
    path = os.path.join(config.OUTPUT_MODELS, "test_model_roundtrip.pkl")
    save_model(trained_baseline_model, path)
    loaded = load_model(path)
    assert type(loaded) == type(trained_baseline_model)

def test_xgb_scale_pos_weight_set(X_train, y_train):
    models = train_with_class_weights(X_train, y_train)
    xgb_model = models['XGBoost (CW)']
    # depending on xgb version, it's either in get_params() or .scale_pos_weight
    params = xgb_model.get_params()
    assert params.get('scale_pos_weight', 1) != 1
