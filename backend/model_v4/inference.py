
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.block(x))

class WideResNet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        dim = 1024
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x): return self.net(x)

class EmotionPredictor:
    def __init__(self, weights_path='resnet_weights.pth', components_path='ml_components.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = joblib.load(components_path)
        self.scaler, self.poly, self.gb_model = data['scaler'], data['poly'], data['gb_model']
        
        self.resnet = WideResNet(data['input_dim'], 4).to(self.device)
        self.resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.resnet.eval()
        self.classes = ['Boring', 'Calm', 'Horror', 'Funny']

    def predict(self, X_raw, threshold=0.65):
        # 1. Feature Prep
        if len(X_raw.shape) == 1: X_raw = X_raw.reshape(1, -1)
        X_proc = self.scaler.transform(self.poly.transform(np.log1p(X_raw)))
        
        # 2. Inference
        with torch.no_grad():
            p_res = torch.softmax(self.resnet(torch.tensor(X_proc, dtype=torch.float32).to(self.device)), 1).cpu().numpy()
        p_gb = self.gb_model.predict_proba(X_proc)
        
        # 3. Ensemble & Threshold
        final_probs = 0.6 * p_res + 0.4 * p_gb
        labels = []
        for i in range(len(final_probs)):
            if np.max(final_probs[i]) < threshold: labels.append("Neutral")
            else: labels.append(self.classes[np.argmax(final_probs[i])])
        return labels, final_probs
