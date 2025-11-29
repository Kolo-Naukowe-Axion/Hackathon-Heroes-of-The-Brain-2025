
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# Definicje Architektury
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
        # Wybór urządzenia (CPU/GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ładowanie komponentów
        if not os.path.exists(weights_path) or not os.path.exists(components_path):
            raise FileNotFoundError("Nie znaleziono plików modelu!")
            
        data = joblib.load(components_path)
        self.scaler = data['scaler']
        self.poly = data['poly']
        self.gb_model = data['gb_model']
        
        # Ładowanie Sieci Neuronowej
        self.resnet = WideResNet(data['input_dim'], 4).to(self.device)
        self.resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.resnet.eval()
        
        self.classes = ['Boring', 'Calm', 'Horror', 'Funny']

    def predict(self, X_raw):
        '''
        Zwraca przewidzianą klasę i prawdopodobieństwa.
        X_raw: Tablica numpy o kształcie (N, 20) z mocami pasm.
        '''
        # Przygotowanie danych
        if len(X_raw.shape) == 1: X_raw = X_raw.reshape(1, -1)
        
        # 1. Log Transform + Polynomial + Scaling
        X_proc = self.scaler.transform(self.poly.transform(np.log1p(X_raw)))
        
        # 2. Predykcja Sieci Neuronowej
        with torch.no_grad():
            p_res = torch.softmax(self.resnet(torch.tensor(X_proc, dtype=torch.float32).to(self.device)), 1).cpu().numpy()
        
        # 3. Predykcja Gradient Boosting
        p_gb = self.gb_model.predict_proba(X_proc)
        
        # 4. Średnia Ważona (Ensemble)
        final_probs = 0.6 * p_res + 0.4 * p_gb
        
        # 5. Wynik
        indices = np.argmax(final_probs, axis=1)
        labels = [self.classes[i] for i in indices]
        
        return labels, final_probs

if __name__ == "__main__":
    # Test
    model = EmotionPredictor()
    dummy_data = np.random.rand(5, 20) * 50 
    labels, _ = model.predict(dummy_data)
    print("Przykładowe predykcje:", labels)
