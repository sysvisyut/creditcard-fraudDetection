import pandas as pd
import os
from src.data_loader import load_data, identify_features

def test_load_data_returns_dataframe(raw_df, monkeypatch):
    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: True)
    monkeypatch.setattr("src.data_loader.pd.read_csv", lambda x: raw_df)
    df = load_data("dummy_path.csv")
    assert isinstance(df, pd.DataFrame)

def test_load_data_correct_columns(raw_df, monkeypatch):
    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: True)
    monkeypatch.setattr("src.data_loader.pd.read_csv", lambda x: raw_df)
    df = load_data("dummy_path.csv")
    expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    for col in expected_cols:
        assert col in df.columns

def test_load_data_no_empty_file(raw_df, monkeypatch):
    monkeypatch.setattr("src.data_loader.os.path.exists", lambda x: True)
    monkeypatch.setattr("src.data_loader.pd.read_csv", lambda x: raw_df)
    df = load_data("dummy_path.csv")
    assert df.shape[0] > 0

def test_identify_features_returns_correct_target(raw_df):
    _, target = identify_features(raw_df)
    assert target == 'Class'

def test_identify_features_excludes_target_from_predictors(raw_df):
    predictors, _ = identify_features(raw_df)
    assert 'Class' not in predictors

def test_class_column_is_binary(raw_df):
    unique_vals = raw_df['Class'].unique()
    assert set(unique_vals).issubset({0, 1})
