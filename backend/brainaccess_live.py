import time
import numpy as np
import torch
import joblib
import pandas as pd
import threading
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
from collections import deque
import torch.nn as nn

# --- 1. KONFIGURACJA ---
MODEL_PATH = 'emotion_model.pth'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'
BUFFER_LENGTH = 250  # ok. 1 sekunda przy 250Hz
FS = 250             # Częstotliwość próbkowania BrainAccess

# Definicja pasm
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# --- 2. DEFINICJA MODELU ---
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

# --- 3. PRZETWARZANIE SYGNAŁU ---
def extract_band_powers(data_buffer, fs):
    data = np.array(data_buffer)
    n_samples, n_channels = data.shape
    features = []
    
    for ch in range(n_channels):
        sig = data[:, ch]
        freqs, psd = welch(sig, fs, nperseg=n_samples)
        
        for band_name, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.sum(psd[idx])
            if band_power <= 0: band_power = 1e-10
            features.append(np.log10(band_power))
            
    return np.array(features).reshape(1, -1)

# --- 4. KLASA DETEKTORA ---
class EmotionDetector:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Stan
        self.current_emotion = "WAITING..."
        self.current_probs = [0.0, 0.0, 0.0] # Neg, Neu, Pos
        self.history = deque(maxlen=30) # Ostatnie 30 odczytów (1 na sek)
        
        # Ładowanie AI
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.le = joblib.load(ENCODER_PATH)
            self.input_dim = self.scaler.mean_.shape[0]
            self.num_classes = len(self.le.classes_)
            
            self.model = SimpleEmotionMLP(self.input_dim, num_classes=self.num_classes)
            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.model.eval()
            print("Model załadowany pomyślnie.")
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            self.model = None

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Detektor uruchomiony w tle.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_data(self):
        with self.lock:
            return {
                "emotion": self.current_emotion,
                "probabilities": self.current_probs, # [Neg, Neu, Pos]
                "history": list(self.history)
            }

    def _run_loop(self):
        print("Szukanie strumienia EEG (LSL)...")
        streams = resolve_byprop('type', 'EEG', timeout=2.0)
        if not streams:
            print("Nie znaleziono strumienia EEG! Przełączanie w tryb MOCK (symulacja).")
            self._run_mock_loop()
            return

        inlet = StreamInlet(streams[0])
        print("Połączono z BrainAccess!")
        
        raw_buffer = deque(maxlen=BUFFER_LENGTH)
        feature_buffer = deque(maxlen=30) # Do adaptacyjnej normalizacji
        prediction_buffer = deque(maxlen=5) # Do wygładzania
        
        while self.running:
            try:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                if chunk:
                    for sample in chunk:
                        raw_buffer.append(sample)
                    
                    if len(raw_buffer) == BUFFER_LENGTH:
                        # 1. Ekstrakcja (Logarytm)
                        features = extract_band_powers(raw_buffer, FS)
                        
                        # Fix wymiarów
                        if features.shape[1] != self.input_dim:
                            features_fixed = np.zeros((1, self.input_dim))
                            min_dim = min(features.shape[1], self.input_dim)
                            features_fixed[:, :min_dim] = features[:, :min_dim]
                            features = features_fixed

                        # 2. Adaptacyjna Normalizacja
                        feature_buffer.append(features[0])
                        if len(feature_buffer) < 10:
                            with self.lock:
                                self.current_emotion = "CALIBRATING..."
                            continue
                            
                        buff_array = np.array(feature_buffer)
                        mean = np.mean(buff_array, axis=0)
                        std = np.std(buff_array, axis=0)
                        std[std == 0] = 1.0
                        features_scaled = (features - mean) / std
                        
                        # 3. Inferencja
                        with torch.no_grad():
                            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                            logits = self.model(input_tensor)
                            probs = torch.softmax(logits, dim=1).numpy()[0]
                        
                        # 4. Wygładzanie
                        prediction_buffer.append(probs)
                        avg_probs = np.mean(prediction_buffer, axis=0)
                        pred_idx = np.argmax(avg_probs)
                        label = self.le.inverse_transform([pred_idx])[0]
                        
                        # Aktualizacja stanu
                        with self.lock:
                            self.current_emotion = label
                            self.current_probs = avg_probs.tolist()
                            self.history.append({
                                "time": time.time(),
                                "probs": avg_probs.tolist(),
                                "label": label
                            })
                            
            except Exception as e:
                print(f"Błąd w pętli: {e}")
                time.sleep(1)

    def _run_mock_loop(self):
        """Symuluje działanie detektora bez urządzenia."""
        import random
        print("Uruchomiono tryb symulacji.")
        
        # Mockowe emocje do cyklicznego przełączania
        mock_emotions = ["Neutral", "Happy", "Sad", "Angry", "Calm"]
        current_mock_idx = 0
        last_switch_time = time.time()
        
        while self.running:
            # Co 5 sekund zmiana emocji
            if time.time() - last_switch_time > 5.0:
                current_mock_idx = (current_mock_idx + 1) % len(mock_emotions)
                last_switch_time = time.time()
            
            emotion = mock_emotions[current_mock_idx]
            
            # Generuj losowe prawdopodobieństwa z przewagą dla obecnej emocji
            probs = [0.1, 0.1, 0.1] # Placeholder, bo model ma 3 klasy (zazwyczaj)
            # Ale w kodzie wyżej jest num_classes=3. 
            # Zakładamy mapowanie: 0->Neg, 1->Neu, 2->Pos (zazwyczaj tak jest w prostych modelach walencji)
            # Ale tutaj `label_encoder` decyduje.
            # W mocku po prostu zwracamy label.
            
            with self.lock:
                self.current_emotion = emotion
                self.current_probs = [0.33, 0.33, 0.33] # Dummy values
                self.history.append({
                    "time": time.time(),
                    "probs": self.current_probs,
                    "label": emotion
                })
            
            time.sleep(0.1) # 10Hz update rate

if __name__ == "__main__":
    # Tryb standalone (testowy)
    detector = EmotionDetector()
    detector.start()
    try:
        while True:
            data = detector.get_data()
            print(f"Stan: {data['emotion']} | Probs: {data['probabilities']}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        detector.stop()
