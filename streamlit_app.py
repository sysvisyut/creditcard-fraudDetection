import streamlit as st
import pandas as pd
import os
import io
import contextlib
from PIL import Image

import config
from src.data_loader import load_data, identify_features
from src.preprocessing import full_preprocessing_pipeline
from src.features import scale_features, select_features, reduce_dimensions
from src.sampling import split_data, apply_smote, apply_oversample, apply_undersample, bias_variance_analysis
from src.models import train_baseline_models, train_sampling_models, train_with_class_weights, train_final_model, save_model
from src.evaluation import evaluate_model, build_master_comparison_table, tune_threshold, evaluate_final_model, compute_clustering_score, generate_master_comparison, generate_business_impact_report
from src.visualization import (plot_class_distribution, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, 
                               plot_sampling_confusion_matrices, plot_sampling_roc_curves, plot_sampling_pr_curves,
                               plot_classweight_confusion_matrices, plot_classweight_roc_curves, plot_classweight_pr_curves,
                               plot_threshold_curves, plot_threshold_confusion_matrices,
                               plot_final_confusion_matrix, plot_final_roc_curve, plot_final_pr_curve, plot_roc_vs_pr_comparison,
                               plot_final_feature_importance, plot_clustering_analysis, plot_master_comparison)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.markdown("""
<style>
/* ===== ANIMATED GRADIENT BACKGROUND ===== */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float1 {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.12; }
    50%      { transform: translate(60px, -40px) scale(1.15); opacity: 0.18; }
}
@keyframes float2 {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.10; }
    50%      { transform: translate(-50px, 30px) scale(1.2); opacity: 0.16; }
}
@keyframes float3 {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.08; }
    50%      { transform: translate(30px, 50px) scale(1.1); opacity: 0.14; }
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 25%, #0a1628 50%, #0f1f33 75%, #0a0e1a 100%);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
}

/* Floating ambient orbs for depth */
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(80px);
}
[data-testid="stAppViewContainer"]::before {
    width: 500px; height: 500px;
    top: -100px; right: -100px;
    background: radial-gradient(circle, #00e5ff15, transparent 70%);
    animation: float1 12s ease-in-out infinite;
}
[data-testid="stAppViewContainer"]::after {
    width: 400px; height: 400px;
    bottom: -80px; left: -80px;
    background: radial-gradient(circle, #ffaa0012, transparent 70%);
    animation: float2 15s ease-in-out infinite;
}

/* Third orb via sidebar pseudo (decorative only) */
[data-testid="stSidebar"]::after {
    content: '';
    position: fixed;
    width: 350px; height: 350px;
    top: 40%; left: 30%;
    border-radius: 50%;
    background: radial-gradient(circle, #7c3aed10, transparent 70%);
    filter: blur(90px);
    pointer-events: none;
    z-index: 0;
    animation: float3 18s ease-in-out infinite;
}

/* ===== TYPOGRAPHY ===== */
h1 {
    color: #00e5ff !important;
    text-shadow: 0 0 12px #00e5ff80, 0 0 30px #00e5ff40;
    font-weight: 800;
    letter-spacing: 0.5px;
}
h2 {
    color: #38bdf8 !important;
    text-shadow: 0 0 6px #38bdf840;
}
h3 {
    color: #34d399 !important;
    text-shadow: 0 0 5px #34d39940;
}

/* ===== DATA TABLES ===== */
div.stDataFrame {
    border: 1px solid #4d8bff30;
    border-radius: 8px;
    box-shadow: 0 0 15px #4d8bff15, 0 0 30px #4d8bff08;
    backdrop-filter: blur(4px);
}

/* ===== TAB BAR ===== */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #111827, #0f1f33, #111827);
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #ffffff08;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    transition: color 0.3s ease, text-shadow 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0;
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    text-shadow: 0 0 8px #00e5ff60;
    border-bottom: 2px solid #00e5ff;
}

/* ===== METRIC CARDS ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827cc, #0f1f33cc);
    border: 1px solid #4d8bff25;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 4px 20px #00000040;
}
[data-testid="stMetricValue"] {
    color: #ffaa00 !important;
    text-shadow: 0 0 6px #ffaa0040;
}

/* ===== EXPANDERS ===== */
details {
    border: 1px solid #ffffff10 !important;
    border-radius: 8px;
    background: #0d1b2a80;
}

/* ===== ALERTS / INFO BOXES ===== */
div[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 4px;
    backdrop-filter: blur(4px);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #4d8bff40; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4d8bff70; }

/* Ensure main content is above the pseudo-element orbs */
[data-testid="stAppViewContainer"] > div { position: relative; z-index: 1; }
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("A premium neon-themed analytics dashboard visualizing our robust machine learning pipeline for fraud classification.")

# Helper functions to cache the heavy lifting
@st.cache_data
def get_data_and_preprocess():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        df = load_data(config.DATA_PATH)
        predictor_cols, target_col = identify_features(df)
        plot_class_distribution(df[target_col])
        df = full_preprocessing_pipeline(df)
        df = scale_features(df)
        selected_features = select_features(df.drop(target_col, axis=1), df[target_col])
        config.SELECTED_FEATURES = selected_features
        X_final = df[selected_features]
        y_final = df[target_col]
        reduce_dimensions(X_final, y_final)
    return X_final, y_final, selected_features, f.getvalue()

@st.cache_data
def load_splits(X_final, y_final):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_final, y_final)
    return X_train, X_val, X_test, y_train, y_val, y_test, f.getvalue()

@st.cache_resource
def compute_sampling_variants(X_train, y_train):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        sampling_variants = {}
        sampling_variants['original'] = (X_train, y_train)
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
        sampling_variants['smote'] = (X_train_sm, y_train_sm)
        X_train_ros, y_train_ros = apply_oversample(X_train, y_train)
        sampling_variants['random_oversample'] = (X_train_ros, y_train_ros)
        X_train_rus, y_train_rus = apply_undersample(X_train, y_train)
        sampling_variants['random_undersample'] = (X_train_rus, y_train_rus)
    return sampling_variants, f.getvalue()

@st.cache_resource
def run_baseline(X_train, y_train, X_val, y_val):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        baseline_models = train_baseline_models(X_train, y_train)
        all_results = []
        y_pred_dict = {}
        y_prob_dict = {}
        for name, model in baseline_models.items():
            res = evaluate_model(model, name, X_val, y_val)
            all_results.append(res)
            y_pred_dict[name] = res['y_pred']
            y_prob_dict[name] = res['y_prob']
        plot_confusion_matrix(y_val, y_pred_dict)
        plot_roc_curve(y_prob_dict, X_val, y_val)
        plot_pr_curve(y_prob_dict, X_val, y_val)
    return baseline_models, all_results, f.getvalue()

@st.cache_resource
def run_sampling(_sampling_variants, X_val, y_val):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        sampling_models = train_sampling_models(_sampling_variants)
        sampling_results = []
        samp_y_pred_dict = {}
        samp_y_prob_dict = {}
        for name, model in sampling_models.items():
            res = evaluate_model(model, name, X_val, y_val)
            sampling_results.append(res)
            samp_y_pred_dict[name] = res['y_pred']
            samp_y_prob_dict[name] = res['y_prob']
        plot_sampling_confusion_matrices(y_val, samp_y_pred_dict)
        plot_sampling_roc_curves(samp_y_prob_dict, X_val, y_val)
        plot_sampling_pr_curves(samp_y_prob_dict, X_val, y_val)
        clean_samp_results = []
        for r in sampling_results:
            clean_samp_results.append({
                'Sampling Method': r['Model'],
                'Accuracy': r['Accuracy'],
                'Precision': r['Precision'],
                'Recall': r['Recall'],
                'F1': r['F1'],
                'ROC-AUC': r['ROC-AUC']
            })
        samp_results_df = pd.DataFrame(clean_samp_results)
    return sampling_models, sampling_results, samp_results_df, f.getvalue()

@st.cache_resource
def run_cost_sensitive(X_train, y_train, X_val, y_val):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        cw_models = train_with_class_weights(X_train, y_train)
        cw_results = []
        cw_y_pred_dict = {}
        cw_y_prob_dict = {}
        for name, model in cw_models.items():
            res = evaluate_model(model, name, X_val, y_val)
            cw_results.append(res)
            cw_y_pred_dict[name] = res['y_pred']
            cw_y_prob_dict[name] = res['y_prob']
        plot_classweight_confusion_matrices(y_val, cw_y_pred_dict)
        plot_classweight_roc_curves(cw_y_prob_dict, X_val, y_val)
        plot_classweight_pr_curves(cw_y_prob_dict, X_val, y_val)
        clean_cw_results = []
        for r in cw_results:
            clean_cw_results.append({
                'Model': r['Model'],
                'class_weight': 'balanced' if 'XGBoost' not in r['Model'] else 'scale_pos_weight',
                'Accuracy': r['Accuracy'],
                'Precision': r['Precision'],
                'Recall': r['Recall'],
                'F1': r['F1'],
                'ROC-AUC': r['ROC-AUC']
            })
        cw_results_df = pd.DataFrame(clean_cw_results)
    return cw_models, cw_results, cw_results_df, f.getvalue()

@st.cache_data
def tune_best_threshold(_cw_models, X_val, y_val):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        best_cw_model = _cw_models['XGBoost (CW)']
        y_prob_best = best_cw_model.predict_proba(X_val)[:, 1]
        threshold_df, optimal_threshold, high_recall_threshold = tune_threshold(best_cw_model, X_val, y_val)
        plot_threshold_curves(threshold_df, default_threshold=0.5, optimal_threshold=optimal_threshold)
        plot_threshold_confusion_matrices(y_val, y_prob_best, optimal_threshold=optimal_threshold, default_threshold=0.5)
    return threshold_df, optimal_threshold, high_recall_threshold, f.getvalue()

@st.cache_resource
def run_final(_cw_models, optimal_threshold, X_test, y_test, selected_features):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        final_model = _cw_models['XGBoost (CW)']
        save_model(final_model, os.path.join(config.OUTPUT_MODELS, "final_fraud_model.pkl"))
        final_eval_results = evaluate_model(final_model, "Best Model + optimal threshold (FINAL)", X_test, y_test, threshold=optimal_threshold)
        plot_final_confusion_matrix(y_test, final_eval_results['y_pred'])
        plot_final_roc_curve(y_test, final_eval_results['y_prob'])
        plot_final_pr_curve(y_test, final_eval_results['y_prob'])
        plot_roc_vs_pr_comparison(y_test, final_eval_results['y_prob'])
        plot_final_feature_importance(final_model, list(X_test.columns))
        sil_score, y_pred_cluster = compute_clustering_score(X_test, y_test)
        plot_clustering_analysis(X_test, y_pred_cluster, y_test)
    return final_eval_results, final_model, f.getvalue()

@st.cache_data
def generate_tables(all_results, sampling_results, cw_results, final_eval_results, optimal_threshold):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        master_df = generate_master_comparison(all_results, sampling_results, cw_results, final_eval_results, optimal_threshold)
        plot_master_comparison(master_df)
    return master_df, f.getvalue()

def display_image(image_file, caption):
    path = os.path.join(config.OUTPUT_PLOTS, image_file)
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, width="stretch", caption=caption)
    else:
        st.warning(f"Could not load image: {path} (make sure the pipeline executed properly)")

with st.spinner("Executing and caching pipeline step 1/8: Preprocessing..."):
    X_final, y_final, selected_features, log_prep = get_data_and_preprocess()

with st.spinner("Executing and caching pipeline step 2/8: Data Splitting..."):
    X_train, X_val, X_test, y_train, y_val, y_test, log_split = load_splits(X_final, y_final)

with st.spinner("Executing and caching pipeline step 3/8: Sampling..."):
    sampling_variants, log_samp_var = compute_sampling_variants(X_train, y_train)

with st.spinner("Executing and caching pipeline step 4/8: Baseline Models..."):
    baseline_models, all_results, log_base = run_baseline(X_train, y_train, X_val, y_val)

with st.spinner("Executing and caching pipeline step 5/8: Sampling Evaluation..."):
    sampling_models, sampling_results, samp_results_df, log_samp_eval = run_sampling(sampling_variants, X_val, y_val)

with st.spinner("Executing and caching pipeline step 6/8: Cost-Sensitive Models..."):
    cw_models, cw_results, cw_results_df, log_cw = run_cost_sensitive(X_train, y_train, X_val, y_val)

with st.spinner("Executing and caching pipeline step 7/8: Threshold Tuning..."):
    threshold_df, optimal_threshold, high_recall_threshold, log_thresh = tune_best_threshold(cw_models, X_val, y_val)

with st.spinner("Executing and caching pipeline step 8/8: Final Evaluation..."):
    final_eval_results, final_model, log_final = run_final(cw_models, optimal_threshold, X_test, y_test, selected_features)
    master_df, log_master = generate_tables(all_results, sampling_results, cw_results, final_eval_results, optimal_threshold)

# ==================
# UI LAYOUT TABS
# ==================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 Data & Prep", 
    "📊 Baselines", 
    "⚖️ Sampling", 
    "💵 Cost-Sensitive", 
    "🎛️ Thresholds", 
    "🏆 Final Impact"
])

with tab1:
    st.header("Exploratory Data Analysis & Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        display_image("1_bar_chart.png", "Absolute Count")
        display_image("3_donut_chart.png", "Share Percentage")
    with col2:
        display_image("4_log_scale_bar_chart.png", "Log Scale Imbalance Visualization")
        display_image("pca_2d_visualization.png", "PCA Dimensionality Reduction") # Assuming it exists from reduce_dimensions
        
    st.subheader("Data Dimensions")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Features", str(len(selected_features)))
    c2.metric("Train Samples", str(X_train.shape[0]))
    c3.metric("Test Samples", str(X_test.shape[0]))

    with st.expander("Show Console Outputs (Data Loading & Preprocessing)", expanded=False):
        st.text(log_prep)
        st.text(log_split)

with tab2:
    st.header("Baseline Models Performance")
    st.info("Baseline models trained on the highly imbalanced dataset without any remedial scaling or modifications.")
    col1, col2 = st.columns(2)
    with col1:
        display_image("baseline_confusion_matrices.png", "Confusion Matrices")
        display_image("baseline_roc_curves.png", "ROC Curve")
    with col2:
        display_image("baseline_pr_curves.png", "Precision-Recall Curve")
        
    # Build cleanly formatted table for Baseline
    st.subheader("Baseline Table")
    base_clean = []
    for r in all_results:
        base_clean.append({
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1': r['F1'],
            'ROC-AUC': r['ROC-AUC']
        })
    st.dataframe(pd.DataFrame(base_clean), width="stretch")

    with st.expander("Show Console Outputs (Baselines)"):
        st.text(log_base)

with tab3:
    st.header("Sampling Strategy Interventions")
    st.success("**Insight:** SMOTE typically improves recall vs undersampling which loses critical data patterns.")
    st.dataframe(samp_results_df, width="stretch")
    
    display_image("sampling_confusion_matrices.png", "Comparisons across Sampling Methods")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image("sampling_roc_curves.png", "ROC Curves")
    with col2:
        display_image("sampling_pr_curves.png", "Precision-Recall Curves")
        
    with st.expander("Show Console Outputs (Sampling Computations)"):
        st.text(log_samp_var)
        st.text(log_samp_eval)

with tab4:
    st.header("Cost-Sensitive Learning (Class Weights)")
    st.warning("**Insight:** Cost-sensitive learning penalizes misclassification of fraud heavily, improving true positive capture without the need to generate synthetic data points.")
    st.dataframe(cw_results_df, width="stretch")
    
    display_image("classweight_confusion_matrices.png", "Confusion Matrices under Class Weighting")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image("classweight_roc_curves.png", "Cost-Sensitive ROC Curve")
    with col2:
        display_image("classweight_pr_curves.png", "Cost-Sensitive PR Curve")

    with st.expander("Show Console Outputs (Class Weights)"):
        st.text(log_cw)

with tab5:
    st.header("Decision Threshold Tuning")
    st.info('**Insight:** "In fraud detection, recall is critical — missing a fraud costs more than a false alarm. Lowering the decision threshold generally increases recall but reduces precision."')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_thresholds = [0.3, 0.4, 0.5, optimal_threshold, high_recall_threshold]
        subset_df = threshold_df[threshold_df['Threshold'].isin(target_thresholds)].drop_duplicates(subset=['Threshold'])
        st.dataframe(subset_df[['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1']], width="stretch")
        st.metric("Optimal F1 Threshold", f"{optimal_threshold:.2f}")
        st.metric("High Recall Threshold", f"{high_recall_threshold:.2f}")

    with col2:
        display_image("threshold_tuning_curves.png", "Impact of threshold on Precision/Recall/F1")

    display_image("threshold_confusion_matrices.png", "Default VS Optimal Threshold Comparison")
    
    with st.expander("Show Console Outputs (Thresholding)"):
        st.text(log_thresh)

with tab6:
    st.header("Final Model & Business Impact Evaluation")
    st.markdown(f"**Final Chosen Configuration:** `XGBoost with Class Weights` using Threshold = `{optimal_threshold}`")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image("final_confusion_matrix.png", "Final Unseen Data Confusion Matrix")
        display_image("roc_vs_pr_comparison.png", "Final ROC vs PR Metrics")
    with col2:
        display_image("final_feature_importance.png", "Decision Drivers (Feature Importance)")
        display_image("clustering_analysis.png", "PCA Clusters")
        
    st.subheader("Master Comparison Summary")
    st.dataframe(master_df, width="stretch")
    display_image("master_model_comparison.png", "Ultimate Aggregated Metric Chart")
    
    st.markdown("---")
    st.subheader("Business Impact Report")
    
    # We generated the business impact via stdout, so let's extract that or just print the logger output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        generate_business_impact_report(y_test, final_eval_results['y_pred'], final_eval_results['y_prob'], optimal_threshold, "XGBoost (SMOTE + CW)")
    report_text = f.getvalue()
    st.code(report_text, language="markdown")
    
    with st.expander("Show Complete Final Console Output Validation Logs Context"):
        st.text(log_final)
        st.text(log_master)
