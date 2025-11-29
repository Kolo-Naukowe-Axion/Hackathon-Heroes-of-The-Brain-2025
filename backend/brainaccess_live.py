import time
import numpy as np
import torch
import joblib
import threading
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch
from collections import deque
import torch.nn as nn

# --- 1. KONFIGURACJA ---
MODEL_PATH = 'model_v5/resnet_weights.pth'
COMPONENTS_PATH = 'model_v5/ml_components.pkl'
BUFFER_LENGTH = 250  # ok. 1 sekunda przy 250Hz
FS = 250             # Częstotliwość próbkowania BrainAccess
CONFIDENCE_THRESHOLD = 0.65 # Próg pewności dla emocji (dopasowany do model_v5)

# Definicja pasm
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Mapowanie klas modelu do emocji frontendu
# Model outputs: ['Boring', 'Calm', 'Horror', 'Funny']
# Frontend expects: neutral, calm, happy, sad, angry
MODEL_CLASSES = ['Boring', 'Calm', 'Horror', 'Funny']
MODEL_TO_FRONTEND_MAP = {
    'Boring': 'neutral',  # Boring -> neutral
    'Calm': 'calm',       # Calm -> calm
    'Horror': 'angry',    # Horror -> angry
    'Funny': 'happy'      # Funny -> happy
}

# --- 2. DEFINICJA MODELU (WideResNet zgodny z model_v5) ---
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.act = nn.GELU()
    def forward(self, x): 
        return self.act(x + self.block(x))

class WideResNet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        dim = 1024
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x): 
        return self.net(x)

# --- 3. PRZETWARZANIE SYGNAŁU ---
def extract_band_powers(data_buffer, fs):
    """
    Extract raw band powers from EEG data.
    Returns raw power values (not log-transformed) as model_v5 expects log1p to be applied later.
    """
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
            # Return raw band power (model_v5 applies log1p in preprocessing)
            features.append(band_power)
            
    return np.array(features).reshape(1, -1)

# --- 4. KLASA DETEKTORA ---
class EmotionDetector:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Stan
        self.current_emotion = "WAITING..."
        self.current_probs = [0.0, 0.0, 0.0, 0.0] 
        self.history = deque(maxlen=30) 
        self.is_mock = False # Flag to indicate if running in simulation mode
        
        # Ładowanie AI
        try:
            print("Ładowanie komponentów ML...")
            components = joblib.load(COMPONENTS_PATH)
            self.scaler = components['scaler']
            self.poly = components['poly']
            self.gb_model = components.get('gb_model', None)  # Ensemble model (opcjonalny)
            
            # Sprawdzenie czy gb_model istnieje (wymagany dla model_v5)
            if self.gb_model is None:
                print("OSTRZEŻENIE: gb_model nie znaleziony w komponentach! Ensemble nie będzie działał.")
            
            # Wymiary
            # Scaler oczekuje wejścia po transformacji wielomianowej (230 cech)
            self.input_dim = self.scaler.mean_.shape[0] 
            self.num_classes = 4
            
            # Wybór urządzenia (CPU/GPU) - zgodnie z model_v5
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Używane urządzenie: {self.device}")
            
            print(f"Inicjalizacja modelu WideResNet (Input: {self.input_dim}, Classes: {self.num_classes})...")
            self.model = WideResNet(input_dim=self.input_dim, num_classes=self.num_classes)
            
            # Ładowanie wag i przeniesienie na odpowiednie urządzenie
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if self.gb_model is not None:
                print("Model załadowany pomyślnie (z ensemble).")
            else:
                print("Model załadowany pomyślnie (bez ensemble).")
            
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.gb_model = None
            self.device = torch.device('cpu')

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
                "probabilities": self.current_probs,
                "history": list(self.history),
                "is_mock": self.is_mock
            }

    def _run_loop(self):
        print("Szukanie strumienia EEG (LSL)...")
        print("Upewnij się, że aplikacja BrainAccess Board jest uruchomiona i streamuje dane.")
        streams = resolve_byprop('type', 'EEG', timeout=5.0) # Increased timeout
        if not streams:
            print("Nie znaleziono strumienia EEG! Przełączanie w tryb MOCK (symulacja).")
            print("Sprawdź czy:")
            print("1. Urządzenie jest włączone.")
            print("2. Aplikacja BrainAccess Board streamuje (LSL).")
            print("3. Jesteś w tej samej sieci (jeśli dotyczy).")
            self.is_mock = True
            self._run_mock_loop()
            return

        inlet = StreamInlet(streams[0])
        print("Połączono z BrainAccess!")
        
        raw_buffer = deque(maxlen=BUFFER_LENGTH)
        prediction_buffer = deque(maxlen=5) # Do wygładzania
        
        while self.running:
            try:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                if chunk:
                    for sample in chunk:
                        raw_buffer.append(sample)
                    
                    if len(raw_buffer) == BUFFER_LENGTH:
                        # Sprawdzenie czy model został załadowany
                        if self.model is None:
                            print("BŁĄD: Model nie został załadowany! Sprawdź logi inicjalizacji.")
                            time.sleep(1)
                            continue
                        
                        # 1. Ekstrakcja cech (20 cech: 4 kanały * 5 pasm)
                        # Zakładamy, że stream ma 4 kanały. Jeśli więcej, weźmiemy pierwsze 4.
                        # Jeśli mniej, to problem.
                        raw_data = np.array(raw_buffer)
                        if raw_data.shape[1] > 4:
                            raw_data = raw_data[:, :4]
                        elif raw_data.shape[1] < 4:
                            print(f"Ostrzeżenie: Otrzymano tylko {raw_data.shape[1]} kanałów, oczekiwano 4.")
                            continue
                        
                        features = extract_band_powers(raw_data, FS) # Shape (1, 20)
                        
                        # 2. Preprocessing zgodny z model_v5: log1p przed poly transform
                        features_log = np.log1p(features)
                        
                        # 3. Transformacja Wielomianowa (20 -> 230)
                        poly_features = self.poly.transform(features_log)
                        
                        # 4. Normalizacja
                        scaled_features = self.scaler.transform(poly_features)
                        
                        # 5. Inferencja z ensemble (jeśli dostępny)
                        with torch.no_grad():
                            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
                            logits = self.model(input_tensor)
                            p_res = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                        # Ensemble z gradient boosting (jeśli dostępny)
                        if self.gb_model is not None:
                            p_gb = self.gb_model.predict_proba(scaled_features)[0]
                            # Ważona kombinacja: 60% ResNet, 40% GB (zgodnie z model_v5)
                            final_probs = 0.6 * p_res + 0.4 * p_gb
                        else:
                            # Fallback: tylko ResNet jeśli gb_model nie jest dostępny
                            final_probs = p_res
                        
                        # 6. Wygładzanie
                        prediction_buffer.append(final_probs)
                        avg_probs = np.mean(prediction_buffer, axis=0)
                        pred_idx = np.argmax(avg_probs)
                        
                        # 7. Mapowanie na etykietę frontendu
                        # Jeśli pewność jest zbyt niska, wracamy do neutralnego
                        if avg_probs[pred_idx] < CONFIDENCE_THRESHOLD:
                            label = "neutral"
                        else:
                            model_label = MODEL_CLASSES[pred_idx]
                            label = MODEL_TO_FRONTEND_MAP.get(model_label, "neutral")
                        
                        # Logowanie dla debugowania
                        # print(f"Model Pred: {model_label} -> Frontend: {label} | Probs: {avg_probs}")
                        
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
        
        # Mockowe emocje do cyklicznego przełączania (zgodne z frontendem)
        mock_emotions = ["neutral", "calm", "happy", "sad", "angry"]
        current_mock_idx = 0
        last_switch_time = time.time()
        
        while self.running:
            # Co 5 sekund zmiana emocji
            if time.time() - last_switch_time > 5.0:
                current_mock_idx = (current_mock_idx + 1) % len(mock_emotions)
                last_switch_time = time.time()
            
            emotion = mock_emotions[current_mock_idx]
            
            # Generuj prawdopodobieństwa zgodne z 4-klasowym modelem
            # Mapowanie: neutral=0, calm=1, happy=2, sad=3, angry=3 (używamy angry jako Horror)
            probs = [0.1] * 4
            if emotion == "neutral":
                probs[0] = 0.7  # Boring
            elif emotion == "calm":
                probs[1] = 0.7  # Calm
            elif emotion == "happy":
                probs[3] = 0.7  # Funny
            elif emotion == "sad":
                probs[0] = 0.7  # Boring (closest to sad)
            elif emotion == "angry":
                probs[2] = 0.7  # Horror
            
            with self.lock:
                self.current_emotion = emotion
                self.current_probs = probs
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
