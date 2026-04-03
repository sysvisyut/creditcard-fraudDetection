import pytest
import pandas as pd
from src.preprocessing import check_missing, handle_outliers, remove_duplicates, full_preprocessing_pipeline

def test_missing_check_returns_dataframe(raw_df):
    df_out = check_missing(raw_df.copy())
    assert isinstance(df_out, pd.DataFrame)

def test_imputation_removes_all_nulls(raw_df):
    df = raw_df.copy()
    df.loc[0, 'Amount'] = None
    df_out = check_missing(df)
    assert df_out.isnull().sum().sum() == 0

def test_outlier_winsorization_clips_amount(raw_df):
    df = raw_df.copy()
    df.loc[0, 'Amount'] = 99999999
    max_amount_before = df['Amount'].quantile(0.99)
    df_out = handle_outliers(df)
    assert df_out['Amount'].max() <= max_amount_before

def test_outlier_winsorization_clips_time(raw_df):
    df = raw_df.copy()
    df.loc[0, 'Time'] = 99999999
    max_time_before = df['Time'].quantile(0.99)
    df_out = handle_outliers(df)
    assert df_out['Time'].max() <= max_time_before

def test_v_columns_untouched_after_outlier_handling(raw_df):
    df_out = handle_outliers(raw_df.copy())
    for i in range(1, 29):
        col = f'V{i}'
        assert (df_out[col] == raw_df[col]).all()

def test_duplicate_removal_reduces_or_equal_rows(raw_df):
    df = raw_df.copy()
    df = pd.concat([df, df.iloc[[0]]]) # duplicate first row
    before_len = len(df)
    df_out = remove_duplicates(df)
    assert len(df_out) < before_len

def test_pipeline_output_is_dataframe(raw_df):
    df_out = full_preprocessing_pipeline(raw_df.copy())
    assert isinstance(df_out, pd.DataFrame)

def test_pipeline_preserves_class_column(clean_df):
    assert 'Class' in clean_df.columns
