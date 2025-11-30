import time
import sys
import numpy as np
import torch
import joblib
import threading
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import scipy.signal
from collections import deque
import torch.nn as nn
import os

# Force immediate output flushing for all print statements
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# --- 1. CONFIGURATION (Matching model_v5) ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_v5', 'resnet_weights.pth')
COMPONENTS_PATH = os.path.join(SCRIPT_DIR, 'model_v5', 'ml_components.pkl')

# EEG Buffer settings - model_v5 uses 250 samples (1 second window)
WINDOW_SIZE = 250         # 1 second at 250Hz (model_v5 requirement)
BUFFER_LENGTH = 250      # Match window size for model_v5
TARGET_FS = 250          # Sampling frequency for model_v5
CONFIDENCE_THRESHOLD = 0.35  # Lower threshold for more sensitive emotion detection

# Prediction smoothing settings
SMOOTHING_WINDOW = 10     # Average last 10 predictions (~1 second at 10Hz)
PREDICTION_INTERVAL = 0.1 # Predict every 100ms (10Hz)

# Channel names (matching model_v5)
TARGET_CHANNELS = ['AF3', 'AF4', 'O1', 'O2']

# Band definitions (matching model_v5 - includes Delta)
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# Model classes (model_v5 output labels)
MODEL_CLASSES = ['Boring', 'Calm', 'Horror', 'Funny']

# Mapping from model_v5 labels to frontend emotion labels
# Frontend expects: neutral, calm, happy, sad, angry
MODEL_TO_FRONTEND_MAP = {
    'Boring': 'neutral',  # Boring maps to neutral
    'Calm': 'calm',       # Calm maps to calm
    'Horror': 'angry',    # Horror maps to angry
    'Funny': 'happy'      # Funny maps to happy
}


# --- 2. MODEL ARCHITECTURE (Matching model_v5) ---
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
    """
    WideResNet architecture for emotion classification (model_v5).
    Input: Processed features (after log, poly, scale)
    Output: 4 classes (Boring, Calm, Horror, Funny)
    """
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        dim = 1024
        self.net = nn.Sequential(
            nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.3),
            ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4), ResidualBlock(dim, 0.4),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x): return self.net(x)


# --- 3. FEATURE EXTRACTION (Matching model_v5) ---
def compute_band_powers(raw_buffer, fs):
    """
    Computes relative band powers for model_v5.
    Converts Raw EEG (250, 4) -> 20 Features (4 channels * 5 bands)
    
    Args:
        raw_buffer: numpy array of shape (n_samples, n_channels), e.g., (250, 4)
                   Expected order: [AF3, AF4, O1, O2]
        fs: sampling frequency (250 Hz)
    
    Returns:
        numpy array of shape (1, 20) - relative band powers
    """
    eps = 1e-10
    # Welch's Periodogram
    freqs, psd = scipy.signal.welch(raw_buffer, fs, nperseg=len(raw_buffer), axis=0)
    total_power = np.sum(psd, axis=0)

    features = []
    # Loop order MUST match training: Channel 1 (D,T,A,B,G), Channel 2...
    for ch_idx in range(raw_buffer.shape[1]):
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if total_power[ch_idx] == 0:
                val = 0
            else:
                # Relative Band Power
                val = np.sum(psd[idx, ch_idx]) / (total_power[ch_idx] + eps)
            features.append(val)

    return np.array(features).reshape(1, -1)


# --- 4. EMOTION DETECTOR CLASS ---
class EmotionDetector:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # State
        self.current_emotion = "WAITING..."
        self.current_probs = [0.0, 0.0, 0.0, 0.0]  # 4 classes
        self.history = deque(maxlen=30)
        
        # Prediction smoothing buffer - stores last N raw predictions for averaging
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
        
        # Data freshness tracking
        self.last_state_update_time = 0
        self.last_prediction_time = 0
        self.last_data_received_time = 0
        self.buffer_fill_rate = 0.0
        self.prediction_rate = 0.0
        self._get_data_call_count = 0
        
        # Load model_v5 components
        try:
            print("=" * 60)
            print("LOADING MODEL V5 COMPONENTS")
            print("=" * 60)
            
            # Check if files exist
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            if not os.path.exists(COMPONENTS_PATH):
                raise FileNotFoundError(f"Components file not found: {COMPONENTS_PATH}")
            
            print(f"Loading ML components from: {COMPONENTS_PATH}")
            components_data = joblib.load(COMPONENTS_PATH)
            self.scaler = components_data['scaler']
            self.poly = components_data['poly']
            self.gb_model = components_data['gb_model']
            self.input_dim = components_data['input_dim']
            print(f"  ✓ Components loaded successfully")
            print(f"  - Scaler type: {type(self.scaler).__name__}")
            print(f"  - Polynomial features: {type(self.poly).__name__}")
            print(f"  - Gradient Boosting model: {type(self.gb_model).__name__}")
            print(f"  - Input dimension: {self.input_dim}")
            
            # Device selection
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Initialize WideResNet model architecture
            print(f"Loading WideResNet model from: {MODEL_PATH}")
            self.resnet = WideResNet(self.input_dim, num_classes=4)
            
            # Load weights
            print(f"  Loading state dict...")
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            print(f"  State dict keys: {list(state_dict.keys())[:5]}... (showing first 5)")
            
            # Verify model architecture matches
            try:
                self.resnet.load_state_dict(state_dict)
                print(f"  ✓ State dict loaded successfully")
            except RuntimeError as e:
                print(f"  ❌ ERROR: Model architecture mismatch!")
                print(f"  Error: {e}")
                print(f"  This means the saved model doesn't match the current architecture.")
                print(f"  Expected model structure:")
                for name, param in self.resnet.named_parameters():
                    print(f"    {name}: {param.shape}")
                raise
            
            self.resnet = self.resnet.to(self.device)
            self.resnet.eval()  # Set to evaluation mode (disables dropout)
            
            print(f"  ✓ Model loaded successfully")
            print(f"  - Input dimension: {self.input_dim} (after log + poly + scale)")
            print(f"  - Output classes: {MODEL_CLASSES}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ CRITICAL ERROR LOADING MODEL V5")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"\nFull traceback:")
            import traceback
            traceback.print_exc()
            print(f"\n{'='*60}")
            print(f"MODEL AND COMPONENTS SET TO None - PREDICTIONS WILL FAIL")
            print(f"{'='*60}\n")
            self.resnet = None
            self.scaler = None
            self.poly = None
            self.gb_model = None
            self.device = torch.device('cpu')
            
            # Don't silently continue - raise the error so it's clear
            # But allow the detector to start so status checks can work
            print("⚠️ WARNING: Detector will start but cannot make predictions!")
            print("   Fix the model loading error and restart the server.")

    def start(self):
        if self.running:
            return
        self.running = True
        print("=" * 80)
        print("STARTING EMOTION DETECTOR - MODEL V5")
        print("=" * 80)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("✓ Detector thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_data(self):
        """Get current emotion state for WebSocket transmission. ONLY REAL EEG DATA."""
        with self.lock:
            current_time = time.time()
            data_age = current_time - self.last_state_update_time if self.last_state_update_time > 0 else 0
            
            # Copy probabilities to ensure we're not sending a reference
            # Convert to native Python floats to ensure JSON serialization works correctly
            if self.current_probs:
                probs_copy = [float(p) for p in self.current_probs]
            else:
                probs_copy = [0.0, 0.0, 0.0, 0.0]
            
            # Log what we're reading (every call for debugging)
            self._get_data_call_count += 1
            prob_str = ', '.join([f'{p:.3f}' for p in probs_copy])
            if self._get_data_call_count <= 5 or self._get_data_call_count % 10 == 0:
                print(f"[get_data #{self._get_data_call_count}] emotion={self.current_emotion}, probs=[{prob_str}], age={data_age:.3f}s, last_update={self.last_state_update_time:.3f}")
                print(f"  → self.current_probs id={id(self.current_probs)}, probs_copy id={id(probs_copy)}")
                print(f"  → self.current_probs values={self.current_probs}")
                print(f"  → probs_copy values={probs_copy}")
            
            # NO MOCK DATA - always real EEG data
            data = {
                "emotion": self.current_emotion,
                "probabilities": probs_copy,
                "history": list(self.history),
                "is_mock": False,  # Always False - no mock data
                "data_age": data_age,
                "last_update_time": self.last_state_update_time,
                "buffer_fill_rate": self.buffer_fill_rate,
                "prediction_rate": self.prediction_rate
            }
            
            return data

    def _run_loop(self):
        """Main loop: connect to LSL stream and process EEG data."""
        print_flush("=" * 80)
        print_flush("EMOTION DETECTOR - SEARCHING FOR EEG STREAM")
        print_flush("=" * 80)
        print_flush("Looking for EEG stream (LSL)...")
        print_flush("Make sure BrainAccess Board is running and streaming.")
        sys.stdout.flush()
        
        # Scan for available LSL streams with longer timeout
        print("Scanning for LSL streams (this may take up to 10 seconds)...")
        print("Make sure BrainAccess Board is running with LSL streaming enabled!")
        all_streams = resolve_streams(wait_time=10.0)
        
        if all_streams:
            print(f"\n✓ Found {len(all_streams)} LSL stream(s) in network:")
            for i, stream in enumerate(all_streams):
                print(f"  Stream {i+1}:")
                print(f"    Name: '{stream.name()}'")
                print(f"    Type: '{stream.type()}'")
                print(f"    Source ID: '{stream.source_id()}'")
                print(f"    Channels: {stream.channel_count()}")
                print(f"    Sampling Rate: {stream.nominal_srate()} Hz")
        else:
            print("\n✗ No LSL streams found in network.")
            print("\nTROUBLESHOOTING:")
            print("  1. Is BrainAccess Board application running?")
            print("  2. Is the device connected to BrainAccess Board?")
            print("  3. Is LSL streaming enabled in BrainAccess Board?")
            print("  4. Check firewall settings (LSL uses UDP broadcast)")
            print("  5. Try running: python diagnose_lsl.py")
        
        # Try to find EEG stream by type (case-insensitive search)
        print("\nSearching for EEG stream by type='EEG'...")
        streams = resolve_byprop('type', 'EEG', timeout=5.0)
        
        # If not found, try case-insensitive alternative search
        if not streams:
            print("No stream with type='EEG' found, trying case-insensitive search...")
            if all_streams:
                for stream in all_streams:
                    stream_name = stream.name().lower()
                    stream_type = stream.type().lower()
                    # More flexible matching
                    if ('brainaccess' in stream_name or 
                        'eeg' in stream_type or 
                        'eeg' in stream_name or
                        'electroencephalography' in stream_type or
                        stream.channel_count() >= 4):  # EEG typically has 4+ channels
                        print(f"  → Found potential EEG stream: '{stream.name()}' (type: '{stream.type()}')")
                        streams = [stream]
                        break
            else:
                print("  → No streams available for alternative search.")
        
        if not streams:
            print("\n⚠️ No EEG stream found!")
            print("   Waiting for device to connect...")
            print("   Make sure BrainAccess Board is running with LSL streaming enabled.")
            # Keep trying to connect - no mock fallback
            time.sleep(5.0)
            self._run_loop()  # Retry connection
            return

        print(f"✓ Found stream: {streams[0].name()} (type: {streams[0].type()})")
        
        print("\n" + "="*80)
        print("🎯 BCI CONNECTED - RECEIVING REAL EEG DATA")
        print("="*80)
        print("✓ Connected to BrainAccess EEG stream")
        print("="*80 + "\n")
        
        inlet = StreamInlet(streams[0])
        
        # Initialize connection
        print("Initializing connection...")
        time.sleep(1.0)
        
        # Verify data is being received
        print("Verifying connection - waiting for data...")
        verification_timeout = 8.0
        verification_start = time.time()
        verified = False
        samples_received = 0
        
        while time.time() - verification_start < verification_timeout:
            try:
                chunk, timestamps = inlet.pull_chunk(timeout=0.5)
                if chunk and len(chunk) > 0:
                    samples_received += len(chunk)
                    print(f"  ✓ Received chunk: {len(chunk)} samples")
                    if samples_received >= 1:
                        verified = True
                        print(f"✓ Connection verified! Received {samples_received} samples.")
                        break
            except Exception as e:
                print(f"  Verification error: {e}")
                time.sleep(0.5)
        
        if not verified:
            print(f"⚠️ Stream found but limited data received. Continuing anyway...")
        
        print("\n" + "="*80)
        print("✅ CONNECTED TO BRAINACCESS - RECEIVING LIVE EEG DATA")
        print("="*80 + "\n")
        
        # Data buffers
        raw_buffer = deque(maxlen=BUFFER_LENGTH)
        consecutive_no_data_count = 0
        max_no_data_count = 10
        last_prediction_time = 0
        PREDICTION_INTERVAL = 0.1  # 10 Hz prediction rate
        prediction_count = 0
        total_samples_received = 0
        chunk_count = 0
        
        # Rate tracking
        samples_received_times = deque(maxlen=100)
        prediction_times = deque(maxlen=100)
        
        # Status reporting
        last_status_time = time.time()
        STATUS_INTERVAL = 5.0  # Print status every 5 seconds
        
        def print_status():
            """Print current connection and processing status"""
            current_time = time.time()
            with self.lock:
                data_age = current_time - self.last_data_received_time if self.last_data_received_time > 0 else float('inf')
                prediction_age = current_time - self.last_prediction_time if self.last_prediction_time > 0 else float('inf')
                buffer_rate = self.buffer_fill_rate
                pred_rate = self.prediction_rate
                emotion = self.current_emotion
                probs = self.current_probs
            
            print("\n" + "="*80)
            print("📊 EEG CONNECTION STATUS")
            print("="*80)
            
            # Connection status
            if data_age < 2.0:
                print("🟢 EEG STREAM: CONNECTED (receiving data)")
            elif data_age < 5.0:
                print("🟡 EEG STREAM: WARNING (data age: {:.1f}s)".format(data_age))
            else:
                print("🔴 EEG STREAM: DISCONNECTED (no data for {:.1f}s)".format(data_age))
            
            # Data reception
            print(f"   • Samples received: {total_samples_received}")
            print(f"   • Chunks processed: {chunk_count}")
            print(f"   • Buffer fill: {len(raw_buffer)}/{BUFFER_LENGTH} samples")
            print(f"   • Sample rate: {buffer_rate:.1f} Hz")
            
            # Prediction status
            if prediction_age < 1.0:
                print("🟢 PREDICTIONS: ACTIVE (last: {:.1f}s ago)".format(prediction_age))
            elif prediction_age < 3.0:
                print("🟡 PREDICTIONS: SLOW (last: {:.1f}s ago)".format(prediction_age))
            else:
                print("🔴 PREDICTIONS: INACTIVE (last: {:.1f}s ago)".format(prediction_age))
            
            print(f"   • Total predictions: {prediction_count}")
            print(f"   • Prediction rate: {pred_rate:.1f} Hz")
            
            # Current state
            prob_str = ', '.join([f'{p:.3f}' for p in probs])
            print(f"   • Current emotion: {emotion}")
            print(f"   • Probabilities: [{prob_str}]")
            
            print("="*80 + "\n")
        
        while self.running:
            try:
                chunk, timestamps = inlet.pull_chunk(timeout=1.0)
                
                if chunk:
                    consecutive_no_data_count = 0
                    chunk_count += 1
                    total_samples_received += len(chunk)
                    current_receive_time = time.time()
                    
                    with self.lock:
                        self.last_data_received_time = current_receive_time
                    
                    # Track sample rate
                    for _ in range(len(chunk)):
                        samples_received_times.append(current_receive_time)
                    
                    if len(samples_received_times) >= 2:
                        time_span = samples_received_times[-1] - samples_received_times[0]
                        if time_span > 0:
                            fill_rate = len(samples_received_times) / time_span
                            with self.lock:
                                self.buffer_fill_rate = fill_rate
                    
                    # Print status periodically
                    if time.time() - last_status_time >= STATUS_INTERVAL:
                        print_status()
                        last_status_time = time.time()
                    
                    # Log every 20th chunk
                    if chunk_count % 20 == 0:
                        print(f"\n[BCI DATA] Chunk #{chunk_count}: {len(chunk)} samples, "
                              f"Total: {total_samples_received}, Buffer: {len(raw_buffer)}/{BUFFER_LENGTH}")
                        if len(chunk) > 0:
                            chunk_array = np.array(chunk)
                            print(f"  Stats: min={chunk_array.min():.4f}, max={chunk_array.max():.4f}, "
                                  f"mean={chunk_array.mean():.4f}, std={chunk_array.std():.4f}")
                    
                    # Add samples to buffer
                    for sample in chunk:
                        raw_buffer.append(sample)
                
                # Check if we should make a prediction
                current_time = time.time()
                buffer_has_enough = len(raw_buffer) >= WINDOW_SIZE  # Need at least WINDOW_SIZE samples
                time_for_prediction = (current_time - last_prediction_time) >= PREDICTION_INTERVAL
                
                should_predict = buffer_has_enough and time_for_prediction
                
                if should_predict:
                    prediction_count += 1
                    last_prediction_time = current_time
                    
                    # Track prediction rate
                    prediction_times.append(current_time)
                    if len(prediction_times) >= 2:
                        time_span = prediction_times[-1] - prediction_times[0]
                        if time_span > 0:
                            with self.lock:
                                self.last_prediction_time = current_time
                                self.prediction_rate = len(prediction_times) / time_span
                    
                    # Check if model is loaded
                    if self.resnet is None:
                        print("❌ CRITICAL ERROR: ResNet model is None! Model failed to load during initialization.")
                        print("   Check the startup logs for model loading errors.")
                        print("   The detector cannot make predictions without a model.")
                        time.sleep(5)
                        continue
                    if self.scaler is None:
                        print("❌ CRITICAL ERROR: Scaler is None! Scaler failed to load during initialization.")
                        print("   Check the startup logs for scaler loading errors.")
                        print("   The detector cannot make predictions without a scaler.")
                        time.sleep(5)
                        continue
                    if self.poly is None:
                        print("❌ CRITICAL ERROR: Polynomial transformer is None! Failed to load during initialization.")
                        print("   Check the startup logs for component loading errors.")
                        time.sleep(5)
                        continue
                    if self.gb_model is None:
                        print("❌ CRITICAL ERROR: Gradient Boosting model is None! Failed to load during initialization.")
                        print("   Check the startup logs for component loading errors.")
                        time.sleep(5)
                        continue
                        print("❌ CRITICAL ERROR: Scaler is None! Scaler failed to load during initialization.")
                        print("   Check the startup logs for scaler loading errors.")
                        print("   The detector cannot make predictions without a scaler.")
                        time.sleep(5)
                        continue
                    
                    # Convert buffer to numpy array
                    raw_data = np.array(list(raw_buffer))
                    
                    # Ensure we have exactly WINDOW_SIZE samples for model_v5
                    if len(raw_data) < WINDOW_SIZE:
                        print(f"Warning: Only {len(raw_data)} samples, need {WINDOW_SIZE} for model_v5.")
                        continue
                    
                    # Take the last WINDOW_SIZE samples
                    window_data = raw_data[-WINDOW_SIZE:]
                    
                    # Ensure we have 4 channels
                    if window_data.shape[1] > 4:
                        window_data = window_data[:, :4]
                    elif window_data.shape[1] < 4:
                        print(f"Warning: Only {window_data.shape[1]} channels, expected 4.")
                        continue
                    
                    # Log prediction details
                    print(f"\n{'#'*60}")
                    print(f"# PREDICTION #{prediction_count}")
                    print(f"# Window: {WINDOW_SIZE} samples, {window_data.shape[1]} channels")
                    print(f"# Data range: [{window_data.min():.4f}, {window_data.max():.4f}]")
                    print(f"{'#'*60}")
                    
                    # ===== MODEL V5 PIPELINE =====
                    
                    # Step 1: Compute relative band powers (20 features: 4 channels × 5 bands)
                    X_raw = compute_band_powers(window_data, TARGET_FS)
                    
                    # CRITICAL DEBUG: Check feature validity
                    print(f"\n[PRED #{prediction_count}] MODEL V5 PREPROCESSING:")
                    print(f"  Window data stats: shape={window_data.shape}, range=[{window_data.min():.4f}, {window_data.max():.4f}]")
                    print(f"  Band powers shape: {X_raw.shape}, count: {len(X_raw[0])}")
                    print(f"  Band powers range: [{X_raw.min():.4f}, {X_raw.max():.4f}]")
                    if np.any(np.isnan(X_raw)) or np.any(np.isinf(X_raw)):
                        print(f"  ❌ ERROR: Invalid band powers detected (NaN or Inf)!")
                        continue
                    
                    # Step 2: Preprocessing (Log -> Poly -> Scale)
                    try:
                        X_log = np.log1p(X_raw)
                        X_poly = self.poly.transform(X_log)
                        X_proc = self.scaler.transform(X_poly)
                        print(f"  Processed features shape: {X_proc.shape}, range: [{X_proc.min():.4f}, {X_proc.max():.4f}]")
                    except Exception as e:
                        print(f"  ❌ ERROR in preprocessing: {e}")
                        continue
                    
                    # Step 3: ResNet Prediction
                    with torch.no_grad():
                        tensor_in = torch.tensor(X_proc, dtype=torch.float32).to(self.device)
                        print(f"  Input tensor shape: {tensor_in.shape}, range: [{tensor_in.min():.4f}, {tensor_in.max():.4f}]")
                        
                        resnet_output = self.resnet(tensor_in)
                        p_res = torch.nn.functional.softmax(resnet_output, dim=1).cpu().numpy()[0]
                        print(f"  ResNet probabilities: {p_res}")
                    
                    # Step 4: Gradient Boosting Prediction
                    try:
                        p_gb = self.gb_model.predict_proba(X_proc)[0]
                        print(f"  GB probabilities: {p_gb}")
                    except Exception as e:
                        print(f"  ❌ ERROR in GB prediction: {e}")
                        continue
                    
                    # Step 5: Ensemble (0.6 * ResNet + 0.4 * GB)
                    raw_probs = 0.6 * p_res + 0.4 * p_gb
                    print(f"  Ensemble probabilities: {raw_probs}")
                    
                    # CRITICAL DEBUG: Check model outputs
                    print(f"  Model probabilities (after softmax): {raw_probs}")
                    print(f"  Probabilities sum: {raw_probs.sum():.4f} (should be ~1.0)")
                    if np.all(raw_probs == 0) or np.any(np.isnan(raw_probs)):
                        print(f"  ❌ ERROR: Model probabilities are all zeros or NaN!")
                        print(f"  This is the problem! Model is not working correctly.")
                        continue
                    
                    if np.allclose(raw_probs, [0.25, 0.25, 0.25, 0.25], atol=0.01):
                        print(f"  ⚠️ WARNING: Probabilities are uniform (model might not be trained or loaded correctly)")
                    
                    # ===== PREDICTION SMOOTHING =====
                    # Add raw prediction to smoothing buffer
                    self.prediction_buffer.append(raw_probs.copy())
                    print(f"  Added to smoothing buffer (size: {len(self.prediction_buffer)})")
                    
                    # Calculate smoothed probabilities (average over buffer)
                    if len(self.prediction_buffer) >= 3:  # Need at least 3 predictions
                        buffer_array = np.array(list(self.prediction_buffer))
                        final_probs = np.mean(buffer_array, axis=0)
                        print(f"  Smoothed from {len(self.prediction_buffer)} predictions")
                    else:
                        final_probs = raw_probs
                        print(f"  Using raw (buffer has {len(self.prediction_buffer)} predictions)")
                    
                    # Normalize to ensure sum = 1
                    prob_sum = final_probs.sum()
                    final_probs = final_probs / (prob_sum + 1e-10)
                    print(f"  Final probs after smoothing: {final_probs}")
                    print(f"  Final probs sum: {final_probs.sum():.4f}")
                    
                    if np.all(final_probs == 0) or np.any(np.isnan(final_probs)):
                        print(f"  ❌ ERROR: Final probabilities are all zeros or NaN after smoothing!")
                        continue
                    
                    # Get prediction from smoothed probabilities
                    pred_idx = np.argmax(final_probs)
                    print(f"  Predicted class index: {pred_idx} ({MODEL_CLASSES[pred_idx]})")
                    
                    # ===== END MODEL 2.3 GRU PIPELINE =====
                    
                    # Map to frontend label
                    if final_probs[pred_idx] < CONFIDENCE_THRESHOLD:
                        label = "neutral"
                    else:
                        model_label = MODEL_CLASSES[pred_idx]
                        label = MODEL_TO_FRONTEND_MAP.get(model_label, "neutral")
                    
                    # Log prediction result every 20th prediction (reduce spam)
                    if prediction_count % 20 == 0:
                        raw_prob_str = ', '.join([f'{raw_probs[i]:.2f}' for i in range(4)])
                        smooth_prob_str = ', '.join([f'{MODEL_CLASSES[i]}={final_probs[i]:.3f}' for i in range(4)])
                        print(f"[MODEL #{prediction_count}] {MODEL_CLASSES[pred_idx]} -> {label}")
                        print(f"  Raw: [{raw_prob_str}] | Smoothed: [{smooth_prob_str}]")
                    
                    # Update state - REAL EEG DATA
                    update_timestamp = time.time()
                    
                    # Create a deep copy of probabilities to prevent any reference issues
                    probs_to_store = [float(p) for p in final_probs.tolist()]
                    
                    with self.lock:
                        self.current_emotion = label
                        self.current_probs = probs_to_store  # Store deep copy
                        self.last_state_update_time = update_timestamp
                        self.history.append({
                            "time": update_timestamp,
                            "probs": probs_to_store.copy(),  # Deep copy for history too
                            "label": label
                        })
                    
                    # Log every prediction with what we're storing
                    prob_str = ', '.join([f'{p:.3f}' for p in probs_to_store])
                    print(f"[STATE←EEG #{prediction_count}] {label} | Probs=[{prob_str}]")
                    print(f"  Storing probabilities: {probs_to_store}")
                    
                    # Immediately verify what's stored
                    with self.lock:
                        stored_probs = list(self.current_probs)
                        stored_emotion = self.current_emotion
                    stored_prob_str = ', '.join([f'{p:.3f}' for p in stored_probs])
                    
                    print(f"  Verification - Stored probs: {stored_probs}")
                    print(f"  Verification - Stored emotion: {stored_emotion}")
                    
                    if stored_prob_str != prob_str:
                        print(f"  ❌ MISMATCH! Stored probs differ from what we tried to store!")
                        print(f"  Tried to store: [{prob_str}]")
                        print(f"  Actually stored: [{stored_prob_str}]")
                    else:
                        print(f"  ✓ STORED CORRECTLY: emotion={stored_emotion}, probs=[{stored_prob_str}]")
                    
                    # Check if stored probs are all zeros
                    if np.allclose(stored_probs, [0, 0, 0, 0], atol=1e-6):
                        print(f"  ❌ CRITICAL ERROR: Stored probabilities are all zeros!")
                        print(f"  This is why the frontend is getting 0,0,0,0!")
                    else:
                        print(f"  ✓ Stored probabilities are non-zero: max={max(stored_probs):.4f}")
                
                # Handle no data received
                if not chunk:
                    consecutive_no_data_count += 1
                    
                    # Print status when no data is received
                    if time.time() - last_status_time >= STATUS_INTERVAL:
                        print_status()
                        last_status_time = time.time()
                    
                    if consecutive_no_data_count >= max_no_data_count:
                        print(f"\n⚠️⚠️⚠️ NO DATA RECEIVED FOR {max_no_data_count} SECONDS! ⚠️⚠️⚠️")
                        print(f"   This usually means the EEG device disconnected.")
                        print(f"   Checking stream status...")
                        
                        # Check if stream still exists
                        try:
                            check_streams = resolve_byprop('type', 'EEG', timeout=2.0)
                            if check_streams:
                                print(f"  ✓ Stream still exists in network, but no data flowing.")
                                print(f"  → Check BrainAccess Board connection")
                                print(f"  → Verify device is powered on")
                                consecutive_no_data_count = 0
                                continue
                            else:
                                print(f"  ✗ Stream lost from network - attempting to reconnect...")
                                time.sleep(2.0)
                                self._run_loop()  # Retry connection
                                return
                        except Exception as e:
                            print(f"  Error checking stream: {e}")
                            time.sleep(2.0)
                            self._run_loop()  # Retry connection
                            return
                            
            except Exception as e:
                print(f"Error in loop: {e}")
                import traceback
                traceback.print_exc()
                consecutive_no_data_count += 1
                if consecutive_no_data_count >= max_no_data_count:
                    print("Too many errors, attempting to reconnect...")
                    time.sleep(2.0)
                    self._run_loop()  # Retry connection
                    return
                time.sleep(1)

if __name__ == "__main__":
    # Standalone test mode
    detector = EmotionDetector()
    detector.start()
    try:
        while True:
            data = detector.get_data()
            print(f"State: {data['emotion']} | Probs: {[f'{p:.3f}' for p in data['probabilities']]}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        detector.stop()
