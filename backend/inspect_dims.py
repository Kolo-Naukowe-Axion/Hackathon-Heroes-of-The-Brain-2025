import joblib
import sys
import numpy as np

try:
    data = joblib.load('/Users/iwosmura/axion antigra/backend/model_v2/ml_components.pkl')
    print("--- Dimensions ---")
    if 'scaler' in data:
        print(f"Scaler input dim: {data['scaler'].mean_.shape[0]}")
    if 'poly' in data:
        print(f"Poly degree: {data['poly'].degree}")
        # Create dummy data to check output dim
        dummy = np.zeros((1, data['scaler'].mean_.shape[0]))
        poly_out = data['poly'].transform(dummy)
        print(f"Poly output dim: {poly_out.shape[1]}")
    if 'input_dim' in data:
        print(f"Saved input_dim: {data['input_dim']}")
        
except Exception as e:
    print(f"Error: {e}")
