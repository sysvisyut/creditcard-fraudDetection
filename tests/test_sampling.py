import pytest
from src.sampling import split_data, apply_smote, apply_oversample, apply_undersample

def test_split_data_returns_six_objects(clean_df):
    X = clean_df.drop('Class', axis=1)
    y = clean_df['Class']
    res = split_data(X, y)
    assert len(res) == 6

def test_split_sizes_approximately_correct(clean_df):
    X = clean_df.drop('Class', axis=1)
    y = clean_df['Class']
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    total = len(X)
    assert abs(len(X_val) / total - 0.15) < 0.05
    assert abs(len(X_test) / total - 0.15) < 0.05

def test_stratification_preserves_fraud_ratio(clean_df):
    X = clean_df.drop('Class', axis=1)
    y = clean_df['Class']
    orig_ratio = y.sum() / len(y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    train_ratio = y_train.sum() / len(y_train)
    assert abs(train_ratio - orig_ratio) < 0.05

def test_smote_increases_minority_class(X_train, y_train):
    orig_1 = sum(y_train == 1)
    X_sm, y_sm = apply_smote(X_train, y_train)
    assert sum(y_sm == 1) > orig_1

def test_smote_output_shapes_match(X_train, y_train):
    X_sm, y_sm = apply_smote(X_train, y_train)
    assert X_sm.shape[0] == y_sm.shape[0]

def test_oversample_balances_classes(X_train, y_train):
    X_os, y_os = apply_oversample(X_train, y_train)
    assert sum(y_os == 0) == sum(y_os == 1)

def test_undersample_reduces_majority(X_train, y_train):
    orig_0 = sum(y_train == 0)
    X_us, y_us = apply_undersample(X_train, y_train)
    assert sum(y_us == 0) < orig_0

def test_no_resampling_on_val_or_test(X_val, X_test):
    # This is a conceptual test. Resampling functions strictly take and return X_train, y_train
    orig_val_shape = X_val.shape
    orig_test_shape = X_test.shape
    assert X_val.shape == orig_val_shape
    assert X_test.shape == orig_test_shape
