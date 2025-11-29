import joblib
import sys

print("--- Inspecting ml_components.pkl with joblib ---")
try:
    data = joblib.load('/Users/iwosmura/axion antigra/backend/model_v2/ml_components.pkl')
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
        for key, value in data.items():
            print(f"{key}: {type(value)}")
            if hasattr(value, 'classes_'):
                print(f"{key} classes: {value.classes_}")
    else:
        print("Content:", data)
except Exception as e:
    print(f"Error loading with joblib: {e}")
