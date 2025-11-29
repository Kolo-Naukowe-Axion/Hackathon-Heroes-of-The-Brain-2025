import time
import sys
import numpy as np
import torch
import joblib
import threading
from pylsl import StreamInlet, resolve_byprop, resolve_streams
from scipy.signal import welch
from collections import deque
import torch.nn as nn

# Force immediate output flushing for all print statements
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

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
        
        # CRITICAL: Add timestamp to track data freshness
        self.last_state_update_time = 0
        self.last_prediction_time = 0
        self.last_data_received_time = 0
        self.buffer_fill_rate = 0.0  # Samples per second
        self.prediction_rate = 0.0  # Predictions per second
        
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
        print("="*80)
        print("STARTING EMOTION DETECTOR - ALL LOGGING ENABLED")
        print("="*80)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("✓ Detektor uruchomiony w tle. Thread started.")
        print("="*80)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_data(self):
        with self.lock:
            current_time = time.time()
            data_age = current_time - self.last_state_update_time
            
            # Add diagnostic info including freshness metrics
            data = {
                "emotion": self.current_emotion,
                "probabilities": self.current_probs,
                "history": list(self.history),
                "is_mock": self.is_mock,
                "data_age": data_age,  # How old is this data in seconds
                "last_update_time": self.last_state_update_time,
                "buffer_fill_rate": self.buffer_fill_rate,
                "prediction_rate": self.prediction_rate
            }
            
            # Verify probabilities are valid
            if len(self.current_probs) == 4:
                prob_sum = sum(self.current_probs)
                if abs(prob_sum - 1.0) > 0.01:  # Should sum to ~1.0
                    print(f"[WARNING] Probabilities sum to {prob_sum:.4f}, expected ~1.0")
            
            # Warn if data is stale
            if data_age > 1.0 and self.last_state_update_time > 0:
                print(f"[STALE DATA WARNING] Data is {data_age:.2f}s old - predictions may not be running!")
            
            return data

    def _run_loop(self):
        print_flush("="*80)
        print_flush("EMOTION DETECTOR - STARTING LSL STREAM DETECTION")
        print_flush("="*80)
        print_flush("Szukanie strumienia EEG (LSL)...")
        print_flush("Upewnij się, że aplikacja BrainAccess Board jest uruchomiona i streamuje dane.")
        sys.stdout.flush()
        
        # First, try to find all streams for debugging
        print("Skanowanie dostępnych strumieni LSL...")
        all_streams = resolve_streams(wait_time=2.0)
        if all_streams:
            print(f"Znaleziono {len(all_streams)} strumieni LSL:")
            for i, stream in enumerate(all_streams):
                print(f"  Stream {i+1}: Name='{stream.name()}', Type='{stream.type()}', Source='{stream.source_id()}'")
        else:
            print("Brak strumieni LSL w sieci.")
        
        # Try to find EEG stream by type
        streams = resolve_byprop('type', 'EEG', timeout=3.0)
        
        # If not found by type, try to find any stream with 'BrainAccess' in name
        if not streams:
            print("Nie znaleziono strumienia typu 'EEG', szukam alternatywnych...")
            all_streams = resolve_streams(wait_time=2.0)
            if all_streams:
                for stream in all_streams:
                    stream_name = stream.name().lower()
                    stream_type = stream.type().lower()
                    if 'brainaccess' in stream_name or 'eeg' in stream_type or 'eeg' in stream_name:
                        print(f"Znaleziono alternatywny strumień: {stream.name()} (type: {stream.type()})")
                        streams = [stream]
                        break
        
        if not streams:
            print("Nie znaleziono strumienia EEG! Przełączanie w tryb MOCK (symulacja).")
            print("Sprawdź czy:")
            print("1. Urządzenie jest włączone.")
            print("2. Aplikacja BrainAccess Board streamuje (LSL) - włącz LSL Stream w interfejsie.")
            print("3. Jesteś w tej samej sieci (jeśli dotyczy).")
            self.is_mock = True
            self._run_mock_loop()
            return

        print(f"Znaleziono strumień: {streams[0].name()} (type: {streams[0].type()})")
        inlet = StreamInlet(streams[0])
        
        # Wait a moment for the inlet to fully connect
        print("Inicjalizowanie połączenia...")
        time.sleep(0.5)
        
        # Verify that data is actually being streamed before declaring connection
        print("Weryfikowanie połączenia - oczekiwanie na dane...")
        verification_timeout = 5.0  # Increased timeout to 5 seconds
        verification_start = time.time()
        verified = False
        chunks_received = 0
        
        while time.time() - verification_start < verification_timeout:
            chunk, timestamps = inlet.pull_chunk(timeout=0.5)
            if chunk and len(chunk) > 0:
                chunks_received += len(chunk)
                if chunks_received >= 10:  # Wait for at least 10 samples
                    verified = True
                    print(f"Zweryfikowano połączenie! Otrzymano {chunks_received} próbek.")
                    break
        
        if not verified:
            print(f"UWAGA: Znaleziono strumień '{streams[0].name()}', ale otrzymano tylko {chunks_received} próbek w ciągu {verification_timeout}s!")
            print("Urządzenie może nie streamować danych lub strumień jest pusty.")
            print("Przełączanie w tryb MOCK (symulacja).")
            self.is_mock = True
            self._run_mock_loop()
            return
        
        print("Połączono z BrainAccess i otrzymywanie danych!")
        
        raw_buffer = deque(maxlen=BUFFER_LENGTH)
        prediction_buffer = deque(maxlen=1) # NO smoothing - show raw predictions immediately
        consecutive_no_data_count = 0
        max_no_data_count = 10  # Switch to mock after 10 consecutive timeouts
        last_prediction_time = 0
        PREDICTION_INTERVAL = 0.1  # Make prediction every 0.1 seconds (10 Hz) to match WebSocket
        prediction_count = 0
        last_buffer_hash = None  # Track if buffer content is actually changing
        last_prediction_data_hash = None  # Track if predictions are actually different
        total_samples_received = 0
        chunk_count = 0
        last_sample_display_time = 0
        SAMPLE_DISPLAY_INTERVAL = 1.0  # Show samples every 1 second
        
        # CRITICAL FIX: Track data flow metrics
        samples_received_times = deque(maxlen=100)  # Track sample receipt times
        prediction_times = deque(maxlen=100)  # Track prediction times
        loop_start_time = time.time()
        
        while self.running:
            try:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                samples_added = 0
                
                if chunk:
                    consecutive_no_data_count = 0  # Reset counter on successful data
                    chunk_count += 1
                    total_samples_received += len(chunk)
                    current_receive_time = time.time()
                    
                    # CRITICAL FIX: Update data received timestamp
                    with self.lock:
                        self.last_data_received_time = current_receive_time
                    
                    # Track sample receipt for rate calculation
                    for _ in range(len(chunk)):
                        samples_received_times.append(current_receive_time)
                    
                    # Calculate buffer fill rate (samples per second)
                    if len(samples_received_times) >= 2:
                        time_span = samples_received_times[-1] - samples_received_times[0]
                        if time_span > 0:
                            fill_rate = len(samples_received_times) / time_span
                            with self.lock:
                                self.buffer_fill_rate = fill_rate
                    
                    # Log chunk details
                    samples_before = len(raw_buffer)
                    
                    # ALWAYS display live BCI data for debugging
                    current_time_display = time.time()
                    
                    # Always show every chunk now
                    print(f"\n{'='*80}")
                    print(f"[BCI DATA - LIVE] Chunk #{chunk_count} received at {time.strftime('%H:%M:%S', time.localtime())}:")
                    print(f"  - Samples in chunk: {len(chunk)}")
                    print(f"  - Timestamps: {len(timestamps)} timestamps")
                    print(f"  - Total samples received: {total_samples_received}")
                    
                    # Show first and last sample from chunk
                    if len(chunk) > 0:
                        first_sample = chunk[0]
                        last_sample = chunk[-1] if len(chunk) > 1 else first_sample
                        
                        print(f"  - First sample (ch {len(first_sample)} channels): {[f'{v:.4f}' for v in first_sample[:8]]}{'...' if len(first_sample) > 8 else ''}")
                        if len(chunk) > 1:
                            print(f"  - Last sample: {[f'{v:.4f}' for v in last_sample[:8]]}{'...' if len(last_sample) > 8 else ''}")
                        
                        # Show statistics
                        chunk_array = np.array(chunk)
                        print(f"  - Data stats: min={chunk_array.min():.4f}, max={chunk_array.max():.4f}, "
                              f"mean={chunk_array.mean():.4f}, std={chunk_array.std():.4f}")
                        
                        # Show channel-wise stats if multiple channels
                        if chunk_array.ndim == 2 and chunk_array.shape[1] > 1:
                            print(f"  - Channel-wise stats:")
                            for ch in range(min(chunk_array.shape[1], 8)):  # Show first 8 channels
                                ch_data = chunk_array[:, ch]
                                print(f"    Ch{ch}: min={ch_data.min():.4f}, max={ch_data.max():.4f}, "
                                      f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")
                    
                    # Show timestamps if available
                    if timestamps and len(timestamps) > 0:
                        if len(timestamps) == 1:
                            print(f"  - Timestamp: {timestamps[0]:.6f}")
                        else:
                            time_diff = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
                            sample_rate_est = len(timestamps) / time_diff if time_diff > 0 else 0
                            print(f"  - Timestamps: {timestamps[0]:.6f} to {timestamps[-1]:.6f} "
                                  f"(span: {time_diff:.4f}s, est. rate: {sample_rate_est:.1f} Hz)")
                    
                    # Show ALL samples in chunk (limit to reasonable amount)
                    print(f"  - Raw EEG samples (showing first 20 of {len(chunk)} samples):")
                    for i, sample in enumerate(chunk[:20]):  # Show first 20 samples
                        sample_str = [f'{v:8.4f}' for v in sample[:4]]  # First 4 channels
                        print(f"    Sample {i:3d}: {sample_str}")
                    if len(chunk) > 20:
                        print(f"    ... ({len(chunk) - 20} more samples)")
                    
                    print(f"{'='*80}\n")
                    
                    for sample in chunk:
                        raw_buffer.append(sample)
                    
                    samples_added = len(raw_buffer) - samples_before
                
                # CRITICAL FIX: Check for prediction OUTSIDE chunk check
                # This ensures we predict even if no new chunk arrived
                current_time = time.time()
                buffer_is_full = len(raw_buffer) >= BUFFER_LENGTH
                buffer_nearly_full = len(raw_buffer) >= (BUFFER_LENGTH - 20)  # Allow 20 sample tolerance
                time_for_prediction = (current_time - last_prediction_time) >= PREDICTION_INTERVAL
                
                # Calculate buffer health
                buffer_fill_percentage = (len(raw_buffer) / BUFFER_LENGTH) * 100
                
                # Predict if buffer is full and enough time has passed
                should_predict = buffer_is_full and time_for_prediction
                
                # SOLID FIX: Also predict if buffer is nearly full AND we have some data
                # This prevents waiting forever for perfect buffer fill
                if not should_predict and buffer_nearly_full and time_for_prediction and len(raw_buffer) >= 200:
                    should_predict = True
                    print(f"[BUFFER TOLERANCE] Predicting with buffer size {len(raw_buffer)}/{BUFFER_LENGTH} ({buffer_fill_percentage:.1f}% full)")
                
                # SOLID FIX: Log buffer health periodically
                if chunk_count % 50 == 0 and len(raw_buffer) > 0:
                    time_since_last_pred = current_time - last_prediction_time
                    print(f"[BUFFER HEALTH] Size: {len(raw_buffer)}/{BUFFER_LENGTH} ({buffer_fill_percentage:.1f}%) | "
                          f"Fill rate: {self.buffer_fill_rate:.1f} Hz | "
                          f"Time since last pred: {time_since_last_pred:.3f}s | "
                          f"Predictions/min: {self.prediction_rate * 60:.1f}")
                
                if should_predict:
                        prediction_count += 1
                        last_prediction_time = current_time
                        
                        # CRITICAL FIX: Track prediction rate
                        prediction_times.append(current_time)
                        if len(prediction_times) >= 2:
                            time_span = prediction_times[-1] - prediction_times[0]
                            if time_span > 0:
                                pred_rate = len(prediction_times) / time_span
                                with self.lock:
                                    self.last_prediction_time = current_time
                                    self.prediction_rate = pred_rate
                        
                        print(f"\n{'#'*80}")
                        print(f"# MODEL PREDICTION #{prediction_count} - FEEDING DATA TO MODEL")
                        print(f"{'#'*80}")
                        
                        # Sprawdzenie czy model został załadowany
                        if self.model is None:
                            print("BŁĄD: Model nie został załadowany! Sprawdź logi inicjalizacji.")
                            time.sleep(1)
                            continue
                        
                        # 1. Ekstrakcja cech (20 cech: 4 kanały * 5 pasm)
                        # Convert buffer to numpy array - this creates a fresh copy each time
                        # The buffer automatically maintains the most recent BUFFER_LENGTH samples
                        raw_data = np.array(list(raw_buffer))  # Explicit copy to ensure fresh data
                        
                        # Check if data is actually changing by computing a hash of recent samples
                        recent_samples_hash = hash(tuple(raw_data[-10:].flatten().tolist())) if len(raw_data) >= 10 else None
                        data_is_changing = (recent_samples_hash != last_buffer_hash) if last_buffer_hash is not None else True
                        last_buffer_hash = recent_samples_hash
                        
                        # DIAGNOSTIC: Check if data is completely static (all zeros or constant)
                        data_variance = raw_data.var()
                        data_std = raw_data.std()
                        is_static = data_variance < 1e-10 or data_std < 1e-5
                        
                        if is_static and prediction_count > 3:
                            print(f"[CRITICAL WARNING #{prediction_count}] Data appears STATIC! Variance: {data_variance:.10f}, Std: {data_std:.10f}")
                            print(f"[CRITICAL] This suggests the BCI is not sending real EEG data or device is not properly connected!")
                        
                        # Show current buffer state with recent samples
                        print(f"\n[BUFFER STATE #{prediction_count}]")
                        print(f"  - Buffer size: {len(raw_buffer)}/{BUFFER_LENGTH} samples")
                        print(f"  - Samples added from chunk: {samples_added}")
                        print(f"  - Total samples received: {total_samples_received}")
                        print(f"  - Recent samples (last 3):")
                        if len(raw_buffer) >= 3:
                            for i, sample in enumerate(list(raw_buffer)[-3:]):
                                sample_str = [f'{v:.3f}' for v in sample[:4]]  # First 4 channels
                                print(f"    Sample -{2-i}: {sample_str}{'...' if len(sample) > 4 else ''}")
                        
                        # Debug: log buffer statistics
                        print(f"  - Data stats: range=[{raw_data.min():.4f}, {raw_data.max():.4f}], "
                              f"mean={raw_data.mean():.4f}, std={raw_data.std():.4f}, "
                              f"variance={data_variance:.6f}")
                        print(f"  - Data status: Changing={data_is_changing}, Static={is_static}")
                        
                        # Show channel-wise variance to detect if specific channels are dead
                        if raw_data.shape[1] >= 2:
                            print(f"  - Channel variances:")
                            for ch in range(min(raw_data.shape[1], 8)):
                                ch_var = raw_data[:, ch].var()
                                ch_std = raw_data[:, ch].std()
                                ch_status = "✓" if ch_var > 1e-6 else "✗ DEAD"
                                print(f"    Ch{ch}: var={ch_var:.8f}, std={ch_std:.6f} {ch_status}")
                        
                        if raw_data.shape[1] > 4:
                            raw_data = raw_data[:, :4]
                        elif raw_data.shape[1] < 4:
                            print(f"Ostrzeżenie: Otrzymano tylko {raw_data.shape[1]} kanałów, oczekiwano 4.")
                            continue
                        
                        # Extract features from the raw data (sliding window is automatic via deque)
                        print(f"[MODEL INPUT #{prediction_count}] Extracting features from {raw_data.shape[0]} samples, {raw_data.shape[1]} channels...")
                        features = extract_band_powers(raw_data, FS) # Shape (1, 20)
                        
                        # Debug: log feature statistics
                        print(f"[MODEL INPUT #{prediction_count}] Features extracted (20 band powers):")
                        print(f"  - Shape: {features.shape}")
                        print(f"  - Stats: mean={features.mean():.6f}, std={features.std():.6f}, min={features.min():.6f}, max={features.max():.6f}")
                        print(f"  - Feature values: {features.flatten().tolist()}")
                        
                        # 2. Preprocessing zgodny z model_v5: log1p przed poly transform
                        print(f"[MODEL INPUT #{prediction_count}] Step 1: Applying log1p transform...")
                        features_log = np.log1p(features)
                        print(f"  - Log1p stats: mean={features_log.mean():.6f}, std={features_log.std():.6f}, min={features_log.min():.6f}, max={features_log.max():.6f}")
                        
                        # 3. Transformacja Wielomianowa (20 -> 230)
                        print(f"[MODEL INPUT #{prediction_count}] Step 2: Applying polynomial transform (20 -> 230 features)...")
                        poly_features = self.poly.transform(features_log)
                        print(f"  - Poly shape: {poly_features.shape}, stats: mean={poly_features.mean():.6f}, std={poly_features.std():.6f}")
                        
                        # 4. Normalizacja
                        print(f"[MODEL INPUT #{prediction_count}] Step 3: Applying scaler normalization...")
                        scaled_features = self.scaler.transform(poly_features)
                        print(f"  - Scaled shape: {scaled_features.shape}, stats: mean={scaled_features.mean():.6f}, std={scaled_features.std():.6f}")
                        print(f"[MODEL INPUT #{prediction_count}] ✓ Data prepared and ready to feed to model")
                        
                        # 5. Inferencja z ensemble (jeśli dostępny)
                        print(f"[MODEL RUN #{prediction_count}] Feeding data to neural network...")
                        with torch.no_grad():
                            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
                            print(f"  - Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")
                            logits = self.model(input_tensor)
                            print(f"  - Raw logits: {logits.cpu().numpy()[0].tolist()}")
                            p_res = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            print(f"[MODEL OUTPUT #{prediction_count}] ✓ Neural network returned probabilities:")
                            print(f"  - ResNet probs: Boring={p_res[0]:.4f}, Calm={p_res[1]:.4f}, Horror={p_res[2]:.4f}, Funny={p_res[3]:.4f}")
                        
                        # Ensemble z gradient boosting (jeśli dostępny)
                        if self.gb_model is not None:
                            print(f"[MODEL RUN #{prediction_count}] Running Gradient Boosting ensemble...")
                            p_gb = self.gb_model.predict_proba(scaled_features)[0]
                            print(f"  - GB probs: Boring={p_gb[0]:.4f}, Calm={p_gb[1]:.4f}, Horror={p_gb[2]:.4f}, Funny={p_gb[3]:.4f}")
                            # Ważona kombinacja: 60% ResNet, 40% GB (zgodnie z model_v5)
                            final_probs = 0.6 * p_res + 0.4 * p_gb
                            print(f"[MODEL OUTPUT #{prediction_count}] ✓ Ensemble final probabilities:")
                            print(f"  - Final probs: Boring={final_probs[0]:.4f}, Calm={final_probs[1]:.4f}, Horror={final_probs[2]:.4f}, Funny={final_probs[3]:.4f}")
                        else:
                            # Fallback: tylko ResNet jeśli gb_model nie jest dostępny
                            final_probs = p_res
                            print(f"[MODEL OUTPUT #{prediction_count}] ✓ Using ResNet only (no ensemble)")
                        
                        # Check if prediction is actually different from last one
                        prob_hash = hash(tuple(final_probs.round(6)))  # Round to avoid float precision issues
                        prob_is_changing = (prob_hash != last_prediction_data_hash) if last_prediction_data_hash is not None else True
                        last_prediction_data_hash = prob_hash
                        
                        # Debug: log raw prediction before smoothing
                        raw_prob_str = ' '.join([f'{p:.4f}' for p in final_probs])
                        print(f"[DEBUG #{prediction_count}] Raw model output: [{raw_prob_str}] | Prediction changing: {prob_is_changing}")
                        
                        # 6. Use raw probabilities (no smoothing for now to see changes immediately)
                        avg_probs = final_probs.copy()  # Use raw probabilities directly
                        pred_idx = np.argmax(avg_probs)
                        
                        # Debug: log final probabilities
                        prob_str_final = ' '.join([f'{p:.4f}' for p in avg_probs])
                        print(f"[DEBUG #{prediction_count}] Final probs (no smoothing): [{prob_str_final}]")
                        
                        # 7. Mapowanie na etykietę frontendu
                        # Jeśli pewność jest zbyt niska, wracamy do neutralnego
                        if avg_probs[pred_idx] < CONFIDENCE_THRESHOLD:
                            label = "neutral"
                        else:
                            model_label = MODEL_CLASSES[pred_idx]
                            label = MODEL_TO_FRONTEND_MAP.get(model_label, "neutral")
                        
                        # Final logging
                        prob_str = ' '.join([f'{p:.3f}' for p in avg_probs])
                        print(f"[RESULT #{prediction_count}] {model_label} -> {label} | Final Probs: [{prob_str}] | Confidence: {avg_probs[pred_idx]:.3f}")
                        print("-" * 80)
                        
                        # CRITICAL FIX: Always update state with timestamp to track freshness
                        print(f"[STATE UPDATE #{prediction_count}] Updating internal state...")
                        update_timestamp = time.time()
                        with self.lock:
                            old_probs = self.current_probs.copy() if hasattr(self.current_probs, 'copy') else list(self.current_probs)
                            old_emotion = self.current_emotion
                            
                            self.current_emotion = label
                            self.current_probs = avg_probs.tolist()
                            self.last_state_update_time = update_timestamp  # CRITICAL: Track when state was last updated
                            self.history.append({
                                "time": update_timestamp,
                                "probs": avg_probs.tolist(),
                                "label": label
                            })
                            
                            # Verify update happened
                            probs_changed = not np.allclose(old_probs, self.current_probs, atol=1e-6)
                            emotion_changed = (old_emotion != label)
                            
                            print(f"[STATE UPDATE #{prediction_count}] ✓ State updated at {update_timestamp:.3f}:")
                            print(f"  - Emotion: {old_emotion} -> {label} {'(CHANGED)' if emotion_changed else '(same)'}")
                            print(f"  - Probs: {[f'{p:.4f}' for p in old_probs]} -> {[f'{p:.4f}' for p in self.current_probs]}")
                            print(f"  - Probabilities changed: {probs_changed}")
                            print(f"  - State age: 0.000s (fresh)")
                            
                            if not probs_changed and prediction_count > 1:
                                print(f"[WARNING #{prediction_count}] ⚠ Probabilities did NOT change in state update!")
                            else:
                                print(f"[STATE UPDATE #{prediction_count}] ✓ Probabilities updated successfully")
                        
                        print(f"{'#'*80}")
                        print(f"# END PREDICTION #{prediction_count}")
                        print(f"{'#'*80}\n")
                # SOLID FIX: Handle case when no chunk received - still try to predict if buffer has data
                if not chunk:
                    consecutive_no_data_count += 1
                    
                    # Even without new chunk, try to predict if buffer has data and enough time passed
                    # This ensures continuous predictions even with slow/inconsistent data stream
                    if len(raw_buffer) >= 200:  # If we have most of buffer
                        current_time = time.time()
                        buffer_nearly_full = len(raw_buffer) >= (BUFFER_LENGTH - 20)
                        time_for_prediction = (current_time - last_prediction_time) >= PREDICTION_INTERVAL
                        
                        if buffer_nearly_full and time_for_prediction:
                            # Force prediction even without new chunk
                            print(f"[NO CHUNK PREDICTION] No new chunk, but predicting from existing buffer ({len(raw_buffer)} samples)")
                            # Will predict on next iteration
                    
                    if consecutive_no_data_count >= max_no_data_count:
                        print(f"UWAGA: Brak danych z urządzenia przez {max_no_data_count} sekund!")
                        print("Urządzenie prawdopodobnie zostało wyłączone lub rozłączone.")
                        print("Przełączanie w tryb MOCK (symulacja).")
                        with self.lock:
                            self.is_mock = True
                        self._run_mock_loop()
                        return
                            
            except Exception as e:
                print(f"Błąd w pętli: {e}")
                consecutive_no_data_count += 1
                if consecutive_no_data_count >= max_no_data_count:
                    print(f"UWAGA: Wystąpił błąd i brak danych przez {max_no_data_count} sekund!")
                    print("Przełączanie w tryb MOCK (symulacja).")
                    self.is_mock = True
                    self._run_mock_loop()
                    return
                time.sleep(1)

    def _run_mock_loop(self):
        """Symuluje działanie detektora bez urządzenia. Periodically checks for real connection."""
        import random
        print("Uruchomiono tryb symulacji.")
        print("UWAGA: System będzie okresowo sprawdzał dostępność urządzenia (co 10 sekund).")
        
        # Mockowe emocje do cyklicznego przełączania (zgodne z frontendem)
        mock_emotions = ["neutral", "calm", "happy", "sad", "angry"]
        current_mock_idx = 0
        last_switch_time = time.time()
        last_reconnect_check = time.time()
        reconnect_check_interval = 10.0  # Check every 10 seconds
        found_streams = None  # Track if we found streams to reconnect
        
        while self.running:
            # Periodically check if real device becomes available
            if time.time() - last_reconnect_check > reconnect_check_interval:
                last_reconnect_check = time.time()
                print("Sprawdzanie dostępności urządzenia...")
                found_streams = resolve_byprop('type', 'EEG', timeout=2.0)
                if not found_streams:
                    all_streams = resolve_streams(wait_time=1.0)
                    if all_streams:
                        for stream in all_streams:
                            stream_name = stream.name().lower()
                            stream_type = stream.type().lower()
                            if 'brainaccess' in stream_name or 'eeg' in stream_type or 'eeg' in stream_name:
                                found_streams = [stream]
                                break
                
                if found_streams:
                    print(f"Wykryto urządzenie! Przełączanie z trybu symulacji na rzeczywiste połączenie...")
                    print(f"Strumień: {found_streams[0].name()} (type: {found_streams[0].type()})")
                    # Exit mock loop and start real connection
                    break
            
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
        
        # If we broke out of mock loop, try to connect to real device
        if self.running and found_streams:
            print("Próba połączenia z wykrytym urządzeniem...")
            # Restart the main loop to connect
            self._run_loop()
            return

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
