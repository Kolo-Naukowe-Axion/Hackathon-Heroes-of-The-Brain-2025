
import torch
import torch.nn as nn
import joblib
import os
import sys

# Define ResNetMLP as in backend/resnet_model.py
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.5):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetMLP(nn.Module):
    def __init__(self, input_dim=230, hidden_dim=1024, num_classes=4, num_blocks=3):
        super(ResNetMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)

# Path to weights
weights_path = 'backend/model_v5/resnet_weights.pth'
components_path = 'backend/model_v5/ml_components.pkl'

print(f"Testing model loading from {weights_path}")

if not os.path.exists(weights_path):
    print("Weights file not found!")
    sys.exit(1)

try:
    # Load components to get input dim
    components = joblib.load(components_path)
    scaler = components['scaler']
    input_dim = scaler.mean_.shape[0]
    print(f"Input dim from scaler: {input_dim}")

    # Try to load into ResNetMLP
    model = ResNetMLP(input_dim=input_dim, num_classes=4)
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Check keys
    print("\nKeys in state_dict:")
    # print(state_dict.keys())
    
    print("\nKeys in model:")
    # print(model.state_dict().keys())
    
    model.load_state_dict(state_dict)
    print("\nSUCCESS: Model loaded into ResNetMLP without errors.")
except Exception as e:
    print(f"\nFAILURE: Could not load model into ResNetMLP. Error: {e}")

