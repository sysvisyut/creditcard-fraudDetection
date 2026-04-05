# 💳 Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![Status](https://img.shields.io/badge/Status-Final_Release-brightgreen.svg)

## 📌 Overview

Credit card fraud datasets are notoriously imbalanced, often containing less than 0.2% fraudulent transactions. This project implements a fully modular, production-grade machine learning pipeline that mathematically processes this severe class imbalance through an array of data augmentation and cost-sensitive strategies. 

The pipeline automates data ingestion, imputation, outlier handling, feature selection, bias-variance analysis, sampling techniques, ensemble modeling, decision-threshold tuning, and advanced visual profiling. By mapping raw data into final business impact calculations, the system dynamically proves how adjusting the decision boundaries of cost-sensitive algorithms directly drives net revenue.

## 🚀 Key Features and Enhancements

- **Data Profiling & Preprocessing**: Automated handling of missing interpolations. Protects core dimensionality while performing rigorous outlier Winsorization and duplicate checking on non-anonymized features (like `Amount` and `Time`).
- **Bias-Variance Tradeoff Proving**: Integrated testing using scaled regularizations on Logistic Regression to pinpoint algorithm fitting lines computationally.
- **Strategic Resampling Frameworks**: Side-by-side comparative analysis of:
  - Default Unbalanced
  - `SMOTE` (Synthetic Minority Over-sampling)
  - `RandomOverSampler`
  - `RandomUnderSampler`
- **Cost-Sensitive Learning**: Penalized algorithmic implementations natively passing `class_weight='balanced'` and XGBoost's explicit `scale_pos_weight` directly rectifying minority neglect without requiring synthetic manipulation.
- **Decision Threshold Tuning**: The model does not blindly rely on the default `0.5` binary threshold. It recalculates the optimal operational threshold to balance Recall and Precision tailored uniquely to fraud detection.
- **Business Impact Profiler**: Translates raw ML Metrics (F1, Precision) directly into physical monetary costs comparing False Negative loss against False Positive operational investigation hours.

---

## 🏆 Final Model Evaluation & Metrics

After rigorous competitive evaluation against Baselines and Synthetic techniques, **XGBoost backed by Cost-Sensitive Learning (`scale_pos_weight`)** vastly outperformed alternative architectures.

To maximize fraud detection natively, the optimal decision threshold was analytically tuned down from `0.50` to **`0.35`**. We evaluated exactly on untouched Test Data with the following final verified performance:

| Metric | Score Achieved | Meaning |
|---|---|---|
| **ROC-AUC** | **97.08%** | World-class probabilistic separation between legitimate and fraudulent transactions. |
| **Recall** | **80.28%** | The pipeline successfully flags over 80% of actual fraud events in the system. |
| **F1-Score** | **84.44%** | A highly harmonious balance preventing catastrophic False Alarms from destroying precision. |
| **Precision** | **89.06%** | When the model signals an alarm, it is correct nearly 90% of the time, keeping investigation costs razor-thin. |

### 💼 Business Impact Context
On a blind test subset spanning 42.5K transactions:
- **Potential fraud caught**: ~$7,000+
- **Cost of false alarms**: ~$600-$700 *(investigation costs)*
- **Net operational benefit**: Profitable and highly scalable for deployment.

---

## 📂 Architecture Structure

- **`config.py`**: Centralized environment handling magic variables, hyperparameter anchors, and IO paths.
- **`src/data_loader.py`**: Manages explicit ingest tasks and broad raw target classifications.
- **`src/preprocessing.py`**: Sanitizes data integrity handling interpolations, winsorizations, and systematic duplicate purging.
- **`src/features.py`**: Unifies scaling, extraction, mathematical Feature Importance models, and PCA dimension visualizers.
- **`src/sampling.py`**: Houses stratified K-Splitting logic along with Bias-Variance iterations and 4 structural minority clusterings.
- **`src/models.py`**: Encapsulates specific training loops for Scikit estimators (`LogisticRegression`, `RandomForestClassifier`), XGBoost bindings, and `.pkl` object state saving.
- **`src/evaluation.py`**: Drives computational mapping extracting exact matrices. Drives threshold tuning and full master table generations safely via try/except error capturing.
- **`src/visualization.py`**: Centralized matplotlib infrastructure saving all plots cleanly inside `/outputs/plots/`.

---

## 💻 Getting Started

### 1. Requirements Installation
Ensure you have Python 3.9+ installed and operational. Create a clean virtual environment, then install requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline Orchestrator
The pipeline has been refactored sequentially so it runs entirely from a single trigger point.
```bash
python main.py
```

### 3. Review Generated Artifacts
Once complete, the pipeline will natively save:
1. All generated Data analysis and Decision threshold curves internally inside the `outputs/plots/` map directory.
2. Your serialized trained XGBoost implementation to `outputs/models/final_fraud_model.pkl` ready for immediate API inference.
3. Console printouts of the **Master Comparison Table** tracking every tested structural variant.

---

## 🧪 Testing framework
This repository contains explicit unit tests designed to ensure data is never corrupted during mathematical ingestions.

To ensure modifications remain functionally pure, execute:
```bash
python -m pytest tests/
```
*(All pipeline operations from data scaling to curve plot saving validations are structurally tested).*
