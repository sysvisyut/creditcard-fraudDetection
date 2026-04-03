import pytest
import pandas as pd
import numpy as np
import os
import shutil
from src.preprocessing import full_preprocessing_pipeline
from src.sampling import split_data
from src.models import train_baseline_models
import config

@pytest.fixture(autouse=True, scope="session")
def setup_plots():
    config.OUTPUT_PLOTS = "outputs/tests/plots/"
    config.OUTPUT_MODELS = "outputs/tests/models/"
    os.makedirs(config.OUTPUT_PLOTS, exist_ok=True)
    os.makedirs(config.OUTPUT_MODELS, exist_ok=True)
    yield
    # Cleanup is generally good but we might want to manually inspect the tests sometimes, 
    # so we'll leave it but tests will overwrite.

@pytest.fixture(scope="session")
def raw_df():
    np.random.seed(42)
    n_samples = 200
    y = np.zeros(n_samples, dtype=int)
    y[:40] = 1
    np.random.shuffle(y)
    
    data = {
        'Time': np.random.uniform(0, 100000, n_samples),
        'Amount': np.random.uniform(0, 500, n_samples),
        'Class': y
    }
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
    df = pd.DataFrame(data)
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    return df[cols]

@pytest.fixture(scope="session")
def clean_df(raw_df):
    return full_preprocessing_pipeline(raw_df.copy())

@pytest.fixture(scope="session")
def split_dfs(clean_df):
    X = clean_df.drop('Class', axis=1)
    y = clean_df['Class']
    from src.sampling import split_data
    return split_data(X, y)

@pytest.fixture(scope="session")
def X_train(split_dfs): return split_dfs[0]

@pytest.fixture(scope="session")
def X_val(split_dfs): return split_dfs[1]

@pytest.fixture(scope="session")
def X_test(split_dfs): return split_dfs[2]

@pytest.fixture(scope="session")
def y_train(split_dfs): return split_dfs[3]

@pytest.fixture(scope="session")
def y_val(split_dfs): return split_dfs[4]

@pytest.fixture(scope="session")
def y_test(split_dfs): return split_dfs[5]

@pytest.fixture(scope="session")
def trained_baseline_models_dict(X_train, y_train):
    return train_baseline_models(X_train, y_train)

@pytest.fixture(scope="session")
def trained_baseline_model(trained_baseline_models_dict):
    return trained_baseline_models_dict['Logistic Regression']

@pytest.fixture(scope="session")
def trained_rf_model(trained_baseline_models_dict):
    return trained_baseline_models_dict['Random Forest']
