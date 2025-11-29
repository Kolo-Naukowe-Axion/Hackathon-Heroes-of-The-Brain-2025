import torch
import torch.nn as nn
import joblib
import numpy as np
from collections import deque

# 1. Definicja Architektury (Musi być identyczna jak w treningu!)
class SimpleEmotionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super(SimpleEmotionMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return self.layer3(x)

class EmotionPredictor:
    def __init__(self, model_path, scaler_path, encoder_path):
        print("Ładowanie systemu AI...")
        
        # Ładowanie artefaktów
        self.scaler = joblib.load(scaler_path)
        self.le = joblib.load(encoder_path)
        
        # Przygotowanie modelu
        # Pobieramy wymiar wejściowy ze scalera (liczba cech)
        input_dim = self.scaler.mean_.shape[0]
        num_classes = len(self.le.classes_)
        
        self.model = SimpleEmotionMLP(input_dim, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval() # Tryb ewaluacji (wyłącza dropout)
        
        # Bufor do wygładzania wyników (Smoothing)
        # Przechowuje ostatnie 10 predykcji, żeby wynik nie "skakał"
        self.buffer = deque(maxlen=10)

    def predict(self, raw_data_vector):
        """
        raw_data_vector: numpy array o kształcie (1, n_features)
        """
        # 1. Normalizacja (Kluczowe!)
        # Musimy użyć tego samego scalera co w treningu
        scaled_data = self.scaler.transform(raw_data_vector)
        
        # 2. Konwersja na Tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        
        # 3. Predykcja modelu
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        # 4. Wygładzanie (Średnia krocząca)
        self.buffer.append(probs)
        avg_probs = np.mean(self.buffer, axis=0)
        
        # 5. Wynik końcowy
        predicted_idx = np.argmax(avg_probs)
        label = self.le.inverse_transform([predicted_idx])[0]
        confidence = avg_probs[predicted_idx]
        
        return label, confidence, avg_probs

# --- SYMULACJA DZIAŁANIA (Mock) ---
if __name__ == "__main__":
    # Inicjalizacja
    predictor = EmotionPredictor(
        model_path='emotion_model.pth',
        scaler_path='scaler.pkl',
        encoder_path='label_encoder.pkl'
    )
    
    print("\nSymulacja danych z opaski EEG...")
    
    # Symulujemy 20 próbek danych (normalnie tu byś czytał z Muse LSL)
    # Generujemy losowe dane o odpowiednim wymiarze
    dummy_input_dim = predictor.scaler.mean_.shape[0]
    
    for i in range(10):
        # Symulacja surowych danych (szum)
        dummy_data = np.random.randn(1, dummy_input_dim)
        
        # Predykcja
        emotion, conf, probs = predictor.predict(dummy_data)
        
        print(f"Chwila {i}: Emocja: {emotion} (Pewność: {conf:.2f}) | {probs}")
