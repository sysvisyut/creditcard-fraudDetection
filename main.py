import config
from src.data_loader import load_data, identify_features
from src.preprocessing import full_preprocessing_pipeline
from src.features import scale_features, select_features, reduce_dimensions
from src.sampling import split_data, apply_smote, apply_oversample, apply_undersample, bias_variance_analysis
from src.models import train_baseline_models
from src.evaluation import evaluate_model, build_master_comparison_table
from src.visualization import plot_class_distribution, plot_confusion_matrix, plot_roc_curve, plot_pr_curve

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

if __name__ == "__main__":
    main()
