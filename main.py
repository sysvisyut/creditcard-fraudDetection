import pandas as pd
import config
from src.data_loader import load_data, identify_features
from src.preprocessing import full_preprocessing_pipeline
from src.features import scale_features, select_features, reduce_dimensions
from src.sampling import split_data, apply_smote, apply_oversample, apply_undersample, bias_variance_analysis
from src.models import train_baseline_models, train_sampling_models, train_with_class_weights
from src.evaluation import evaluate_model, build_master_comparison_table, tune_threshold
from src.visualization import (plot_class_distribution, plot_confusion_matrix, plot_roc_curve, plot_pr_curve, 
                               plot_sampling_confusion_matrices, plot_sampling_roc_curves, plot_sampling_pr_curves,
                               plot_classweight_confusion_matrices, plot_classweight_roc_curves, plot_classweight_pr_curves,
                               plot_threshold_curves, plot_threshold_confusion_matrices)

def main():
    # 1. Load Data
    df = load_data(config.DATA_PATH)
    
    # 2. Explore & Identify Target/Features
    predictor_cols, target_col = identify_features(df)
    
    # 3. Class Distribution Visualizations
    plot_class_distribution(df[target_col])
    
    # 4. Preprocess Data
    df = full_preprocessing_pipeline(df)
    
    # 5. Transform & Scale
    df = scale_features(df)
    
    # 6. Feature Selection
    selected_features = select_features(df.drop(target_col, axis=1), df[target_col])
    config.SELECTED_FEATURES = selected_features
    
    X_final = df[selected_features]
    y_final = df[target_col]
    
    # 7. Dimensionality Reduction (PCA Visualization)
    reduce_dimensions(X_final, y_final)
    
    # 8. Data Splitting
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_final, y_final)
    
    # 9. Bias-Variance Analysis
    bias_variance_analysis(X_train, y_train, X_val, y_val)
    
    # 10. Sampling Strategies
    print("\n--- 3. SAMPLING STRATEGIES ---")
    sampling_variants = {}
    print(f"Original Training Class Distribution: {dict(y_train.value_counts())}")
    sampling_variants['original'] = (X_train, y_train)
    
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    sampling_variants['smote'] = (X_train_sm, y_train_sm)
    
    X_train_ros, y_train_ros = apply_oversample(X_train, y_train)
    sampling_variants['random_oversample'] = (X_train_ros, y_train_ros)
    
    X_train_rus, y_train_rus = apply_undersample(X_train, y_train)
    sampling_variants['random_undersample'] = (X_train_rus, y_train_rus)

    # 11. Evaluate Baseline Models
    print("-" * 50)
    print("BASELINE MODEL EVALUATION (NO BALANCING)")
    print("-" * 50)
    
    baseline_models = train_baseline_models(sampling_variants['original'][0], sampling_variants['original'][1])
    
    all_results = []
    y_pred_dict = {}
    y_prob_dict = {}
    
    for name, model in baseline_models.items():
        print(f"\nEvaluating {name}...")
        res = evaluate_model(model, name, X_val, y_val)
        all_results.append(res)
        y_pred_dict[name] = res['y_pred']
        y_prob_dict[name] = res['y_prob']
        
    # Plot baseline matrices and curves
    plot_confusion_matrix(y_val, y_pred_dict)
    plot_roc_curve(y_prob_dict, X_val, y_val)
    plot_pr_curve(y_prob_dict, X_val, y_val)
    
    # Summary
    build_master_comparison_table(all_results)
    
    # 12. Evaluate Sampling Variants
    sampling_models = train_sampling_models(sampling_variants)
    
    sampling_results = []
    samp_y_pred_dict = {}
    samp_y_prob_dict = {}
    
    for name, model in sampling_models.items():
        print(f"\nEvaluating {name}...")
        res = evaluate_model(model, name, X_val, y_val)
        sampling_results.append(res)
        samp_y_pred_dict[name] = res['y_pred']
        samp_y_prob_dict[name] = res['y_prob']
        
    # Plot visualization matrices
    plot_sampling_confusion_matrices(y_val, samp_y_pred_dict)
    plot_sampling_roc_curves(samp_y_prob_dict, X_val, y_val)
    plot_sampling_pr_curves(samp_y_prob_dict, X_val, y_val)
    
    # 13. Comparison Table for Sampling
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
    print("\n--- SAMPLING RESULTS SUMMARY ---")
    print(samp_results_df.to_string(index=False))
    print("\nInsight: SMOTE typically improves recall vs undersampling which loses data.")

    # 14. Cost-Sensitive Learning (Class Weights)
    cw_models = train_with_class_weights(sampling_variants['original'][0], sampling_variants['original'][1])
    
    cw_results = []
    cw_y_pred_dict = {}
    cw_y_prob_dict = {}
    
    for name, model in cw_models.items():
        print(f"\nEvaluating {name}...")
        res = evaluate_model(model, name, X_val, y_val)
        cw_results.append(res)
        cw_y_pred_dict[name] = res['y_pred']
        cw_y_prob_dict[name] = res['y_prob']
        
    # Plot visualization matrices
    plot_classweight_confusion_matrices(y_val, cw_y_pred_dict)
    plot_classweight_roc_curves(cw_y_prob_dict, X_val, y_val)
    plot_classweight_pr_curves(cw_y_prob_dict, X_val, y_val)
    
    # 15. Comparison Table for Class Weights
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
    print("\n--- COST-SENSITIVE RESULTS SUMMARY ---")
    print(cw_results_df.to_string(index=False))
    print("\nInsight: Cost-sensitive learning penalizes misclassification of fraud more heavily, improving recall without generating synthetic data.")

    # 16. Decision Threshold Tuning
    best_rf_cw_model = cw_models['Random Forest (CW)']
    y_prob_best = best_rf_cw_model.predict_proba(X_val)[:, 1]
    
    threshold_df, optimal_threshold, high_recall_threshold = tune_threshold(best_rf_cw_model, X_val, y_val)
    
    # Visualizations
    plot_threshold_curves(threshold_df, default_threshold=0.5, optimal_threshold=optimal_threshold)
    plot_threshold_confusion_matrices(y_val, y_prob_best, optimal_threshold=optimal_threshold, default_threshold=0.5)
    
    # Re-evaluate
    print(f"\nRe-evaluating Optimized Model at Threshold = {optimal_threshold}")
    evaluate_model(best_rf_cw_model, "Random Forest (CW) Optimized", X_val, y_val, threshold=optimal_threshold)
    
    # 17. Subset DataFrame Outputs
    target_thresholds = [0.3, 0.4, 0.5, optimal_threshold, high_recall_threshold]
    # filter and drop duplicates if thresholds overlap exactly
    subset_df = threshold_df[threshold_df['Threshold'].isin(target_thresholds)].drop_duplicates(subset=['Threshold'])
    
    # Explicitly pull accuracy, precision, recall, f1 dynamically 
    print("\n--- THRESHOLD COMPARISON TABLE ---")
    print(subset_df[['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1']].to_string(index=False))
    
    print('\nInsight: "In fraud detection, recall is critical — missing a fraud costs more than a false alarm. Lowering threshold increases recall but reduces precision."')

if __name__ == "__main__":
    main()
