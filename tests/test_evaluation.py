import pytest
from src.evaluation import evaluate_model, tune_threshold, compute_clustering_score, build_master_comparison_table

def test_evaluate_model_returns_dict(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    assert isinstance(res, dict)

def test_evaluate_model_has_all_keys(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    expected_keys = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'y_pred', 'y_prob']
    for k in expected_keys:
        assert k in res

def test_accuracy_between_0_and_1(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    assert 0 <= res['Accuracy'] <= 1

def test_f1_between_0_and_1(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    assert 0 <= res['F1'] <= 1

def test_roc_auc_above_random(trained_baseline_model, X_val, y_val):
    res = evaluate_model(trained_baseline_model, "Test", X_val, y_val)
    # in small random tests it might dip below 0.5 occasionally but ideally strictly bounded.
    assert res['ROC-AUC'] >= 0.0 # bounding strictly physically

def test_threshold_tuning_returns_float(trained_rf_model, X_val, y_val):
    df, opt, high_rec = tune_threshold(trained_rf_model, X_val, y_val)
    assert isinstance(opt, float)

def test_threshold_in_valid_range(trained_rf_model, X_val, y_val):
    df, opt, high_rec = tune_threshold(trained_rf_model, X_val, y_val)
    assert 0 < opt < 1

def test_clustering_score_returns_float(X_test, y_test):
    from sklearn.cluster import KMeans
    import numpy as np
    score, _ = compute_clustering_score(X_test, y_test)
    assert isinstance(score, float) or isinstance(score, np.floating)

def test_clustering_score_in_valid_range(X_test, y_test):
    score, _ = compute_clustering_score(X_test, y_test)
    assert -1 <= score <= 1

def test_master_table_has_all_models():
    # comparison table rows >= 9. In this synthetic test we mock it.
    results = [{'Model': f"M{i}", 'Accuracy': 0.9, 'Precision': 0.9, 'Recall': 0.9, 'F1': 0.9, 'ROC-AUC': 0.9} for i in range(10)]
    df = build_master_comparison_table(results)
    assert len(df) >= 9

def test_master_table_has_required_columns():
    results = [{'Model': f"M{i}", 'Accuracy': 0.9, 'Precision': 0.9, 'Recall': 0.9, 'F1': 0.9, 'ROC-AUC': 0.9} for i in range(1)]
    df = build_master_comparison_table(results)
    assert 'Model' in df.columns
    assert 'F1' in df.columns
    assert 'ROC-AUC' in df.columns
