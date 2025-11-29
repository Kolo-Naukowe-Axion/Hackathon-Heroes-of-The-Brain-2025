import joblib
import os
import sys

path = 'model_v5/ml_components.pkl'
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

try:
    components = joblib.load(path)
    print(f"Loaded components from {path}")
    print(f"Type: {type(components)}")
    
    if isinstance(components, dict):
        print(f"Keys: {components.keys()}")
        
        if 'scaler' in components:
            scaler = components['scaler']
            print(f"Scaler mean shape: {scaler.mean_.shape}")
            print(f"Scaler scale shape: {scaler.scale_.shape}")
            
        if 'poly' in components:
            poly = components['poly']
            print(f"Poly degree: {poly.degree}")
            print(f"Poly n_output_features: {poly.n_output_features_}")
            
except Exception as e:
    print(f"Error loading components: {e}")
