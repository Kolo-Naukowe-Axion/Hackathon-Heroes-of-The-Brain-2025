import torch
import sys

try:
    weights = torch.load('/Users/iwosmura/axion antigra/backend/model_v2/resnet_weights.pth', map_location='cpu')
    print("--- Weight Shapes ---")
    for key, value in weights.items():
        print(f"{key}: {value.shape}")
except Exception as e:
    print(f"Error loading weights: {e}")
