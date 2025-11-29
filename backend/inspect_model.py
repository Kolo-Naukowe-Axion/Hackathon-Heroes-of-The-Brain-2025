import torch
import joblib
import sys

print("--- Inspecting ml_components.pkl ---")
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

print("\n--- Inspecting resnet_weights.pth ---")
try:
    weights = torch.load('/Users/iwosmura/axion antigra/backend/model_v2/resnet_weights.pth', map_location='cpu')
    print("Type:", type(weights))
    if isinstance(weights, dict):
        print("Keys (first 10):", list(weights.keys())[:10])
        # Check for specific layer names to infer architecture
        if any('conv1' in k for k in weights.keys()):
            print("Found conv1")
        if any('layer1' in k for k in weights.keys()):
            print("Found layer1")
    else:
        print("Content:", weights)
except Exception as e:
    print(f"Error loading weights: {e}")
