import pickle
import pandas as pd
import numpy as np

# These are the exact top 20 features selected during training
REQUIRED_FEATURES = [
    'V17', 'V14', 'V12', 'V16', 'V10', 'V11', 'V7', 'V9', 'V4', 'V18', 
    'V21', 'V26', 'V3', 'V6', 'V8', 'V27', 'V28', 'V2', 'V20', 'V22'
]

# The optimal operating threshold discovered during threshold tuning
OPTIMAL_THRESHOLD = 0.35 

def load_fraud_model(model_path="outputs/models/final_fraud_model.pkl"):
    """Loads the pre-trained final model from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Run main.py first.")
        return None

def predict_fraud(input_data: dict, model):
    """
    Takes a dictionary of input features and predicts if it's fraudulent.
    
    Args:
        input_data (dict): Dictionary mapping feature names (e.g. 'V17') to their values
        model: The trained ML model
    
    Returns:
        dict: Containing the prediction class and the probability
    """
    # 1. Validate inputs to ensure all required features are present
    missing_features = [f for f in REQUIRED_FEATURES if f not in input_data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # 2. Convert to DataFrame in the exact order the model expects
    # We create a single-row DataFrame using the required feature list
    input_df = pd.DataFrame([input_data])[REQUIRED_FEATURES]
    
    # 3. Predict the probability of Fraud (Class 1)
    # The XGBoost model natively outputs probabilities
    fraud_probability = model.predict_proba(input_df)[0, 1]
    
    # 4. Apply our tuned optimal threshold
    is_fraud = fraud_probability >= OPTIMAL_THRESHOLD
    
    return {
        "status": "FRAUDULENT 🚨" if is_fraud else "LEGITIMATE ✅",
        "fraud_probability": round(fraud_probability * 100, 2),
        "threshold_used": OPTIMAL_THRESHOLD
    }

if __name__ == "__main__":
    # --- Example Usage ---
    model = load_fraud_model()
    
    if model:
        print("\n--- Testing Custom Inference ---")
        
        # 1. Define your custom live transaction input data
        my_custom_transaction = {
            'V17': -1.23, 'V14': -4.56, 'V12': -2.10, 'V16': 0.44, 
            'V10': -1.15, 'V11': 2.31,  'V7': 0.12,  'V9': -0.05, 
            'V4': 0.88,   'V18': 0.15,  'V21': 0.22, 'V26': 0.10, 
            'V3': 1.05,   'V6': -0.76,  'V8': 0.04,  'V27': 0.01, 
            'V28': -0.02, 'V2': 0.50,   'V20': 0.11, 'V22': 0.08
        }
        
        # 2. Pass your custom data to the prediction function
        print("\nProcessing My Custom Transaction...")
        custom_result = predict_fraud(my_custom_transaction, model)
        
        # 3. Print the verification output
        print(f"Status Result: {custom_result['status']}")
        print(f"Confidence/Probability: {custom_result['fraud_probability']}%")
        print(f"Threshold applied: {custom_result['threshold_used']}")
        
        print("\nNote: Time and Amount were not required as they were algorithmically dropped during training feature selection!")
        print("-" * 50)
