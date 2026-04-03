# Credit Card Fraud Detection Pipeline

## Overview
A robust machine learning pipeline systematically handling the severe class imbalance and dimensionality characteristics inherent to binary classification on massive enterprise credit datasets. Refactored into a fully modularized, production-grade template structure encapsulating ingestion, cleaning, PCA mappings, oversampling/undersampling techniques, and automated model evaluations.

## Architecture Structure

- **`config.py`**: A centralized environment handling magic variables, hyperparameter anchors, and operational IO paths.
- **`src/data_loader.py`**: Manages explicit ingest tasks and broad raw target classifications.
- **`src/preprocessing.py`**: Sanitizes data integrity handling interpolations, winsorizations, and systematic duplicate purging.
- **`src/features.py`**: Unifies scales and executes mathematical Feature Importance models (Random Forests, Correlation matrices, and PCA dimension visualizers).
- **`src/sampling.py`**: Houses stratified K-Splitting logic along with Bias-Variance iterations and 4 structural implementations of minority re-clusterings (`SMOTE`, `RandomOverSampler`, `RandomUnderSampler`).
- **`src/models.py`**: Isolated encapsulation wrappers mapping explicit local executions for scikit estimators (`LogisticRegression`, `RandomForestClassifier`), weights, and pickle logic.
- **`src/evaluation.py`**: Drives pure computational mapping extracting specific matrices tied identically to `validation` blocks outputting precision/ROCAUC arrays.
- **`src/visualization.py`**: Centralized Matplotlib infrastructure directly saving specific `outputs/plots/` maps avoiding redundant code throughout logic operations.

## Running the Pipeline

1. **Setup Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Execute Full End-to-End Orchestrator**:
```bash
python main.py
```

## Expected Outputs
The pipeline runs linearly scaling through all layers mathematically without manual intervention. Output distributions, tradeoff heatmaps, bar annotations, ROC metrics and specific PR-AUC clusters immediately overwrite cleanly inside `/outputs/plots`. 
Base classification output arrays natively resolve cleanly inside your system's output console sequentially marking model behavior states ending with the definitive evaluation summary.
