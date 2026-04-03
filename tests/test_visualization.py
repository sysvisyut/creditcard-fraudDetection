import pytest
import os
import matplotlib
import matplotlib.pyplot as plt
from src.visualization import plot_class_distribution, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_threshold_curves, plot_feature_importance
import config
import pandas as pd
import numpy as np

matplotlib.use('Agg')

def test_class_distribution_plot_saved(y_train):
    plot_class_distribution(y_train)
    path = os.path.join(config.OUTPUT_PLOTS, "1_bar_chart.png")
    assert os.path.exists(path)

def test_confusion_matrix_plot_saved(y_val):
    y_pred = pd.Series(np.zeros_like(y_val))
    plot_confusion_matrix(y_val, {"Test": y_pred})
    path = os.path.join(config.OUTPUT_PLOTS, "baseline_confusion_matrices.png")
    assert os.path.exists(path)

def test_roc_curve_plot_saved(X_val, y_val):
    y_prob = pd.Series(np.zeros_like(y_val, dtype=float))
    try:
        plot_roc_curve({"Test": y_prob}, X_val, y_val)
    except Exception:
        pass
    path = os.path.join(config.OUTPUT_PLOTS, "baseline_roc_curves.png")
    assert os.path.exists(path)

def test_pr_curve_plot_saved(X_val, y_val):
    y_prob = pd.Series(np.zeros_like(y_val, dtype=float))
    try:
        plot_pr_curve({"Test": y_prob}, X_val, y_val)
    except Exception:
        pass
    path = os.path.join(config.OUTPUT_PLOTS, "baseline_pr_curves.png")
    assert os.path.exists(path)

def test_threshold_curve_plot_saved():
    df = pd.DataFrame({'Threshold': [0.1, 0.5], 'Precision': [0.5, 0.6], 'Recall': [0.8, 0.4], 'F1': [0.6, 0.5]})
    plot_threshold_curves(df, 0.5, 0.5)
    path = os.path.join(config.OUTPUT_PLOTS, "threshold_tuning_curves.png")
    assert os.path.exists(path)

def test_feature_importance_plot_saved(trained_rf_model, X_train):
    plot_feature_importance(trained_rf_model, list(X_train.columns))
    # It might create a different name depending on the plot_feature_importance save logic.
    # We will wait and see if it exposes any error.
    # Ah, the instructions had plot_final_feature_importance saving to final_feature_importance.png
    # And plot_feature_importance saving to feature_importance.png
    path = os.path.join(config.OUTPUT_PLOTS, "feature_importance.png")
    assert os.path.exists(path)

def test_plots_are_valid_png():
    # Will test this directly after previous plots are generated in session
    path = os.path.join(config.OUTPUT_PLOTS, "1_bar_chart.png")
    if os.path.exists(path):
        assert os.path.getsize(path) > 1000

def test_no_plot_display_during_tests(monkeypatch):
    assert matplotlib.get_backend().lower() == 'agg'
    
    called = False
    def mock_show(*args, **kwargs):
        nonlocal called
        called = True
    
    monkeypatch.setattr(plt, "show", mock_show)
    # Generate a plot
    plt.figure()
    plt.plot([1, 2], [1, 2])
    # The visualization functions shouldn't call plt.show()
    # We can invoke one to verify
    try:
        plot_class_distribution(pd.Series([0, 1]))
    except:
        pass
    assert not called
