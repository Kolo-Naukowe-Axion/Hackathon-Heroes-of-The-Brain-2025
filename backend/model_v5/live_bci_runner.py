# -*- coding: utf-8 -*-


import time
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import joblib
import brainaccess
from brainaccess.utils import acquisition

# --- CONFIGURATION ---
TARGET_CHANNELS = ['AF3', 'AF4', 'O1', 'O2']
TARGET_FS = 250
WINDOW_SIZE = 250  # 1 second window
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

# --- 1. DEFINE MODEL ARCHITECTURE (From your inference.py) ---
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

# --- 2. PREDICTOR CLASS (Adapted for Live Input) ---
class LivePredictor:
    def __init__(self, weights_path='resnet_weights.pth', components_path='ml_components.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[System] Loading models on {self.device}...")

        try:
            # Load ML Components (Scaler, Poly, GB Model)
            data = joblib.load(components_path)
            self.scaler = data['scaler']
            self.poly = data['poly']
            self.gb_model = data['gb_model']
            self.input_dim = data['input_dim'] # Should be around 1300 after poly

            # Load ResNet
            self.resnet = WideResNet(self.input_dim, 4).to(self.device)
            self.resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.resnet.eval()

            self.classes = ['Boring', 'Calm', 'Horror', 'Funny']
            print("[System] Models loaded successfully.")

        except FileNotFoundError:
            print("[Error] Missing 'resnet_weights.pth' or 'ml_components.pkl'")
            raise

    def compute_band_powers(self, raw_buffer, fs):
        """
        Converts Raw EEG (250, 4) -> 20 Features (4 channels * 5 bands)
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

    def predict(self, raw_buffer):
        # 1. Feature Extraction (Raw -> 20 Features)
        X_raw = self.compute_band_powers(raw_buffer, TARGET_FS)

        # 2. Preprocessing (Log -> Poly -> Scale)
        # Matches your inference.py logic exactly
        X_proc = self.scaler.transform(self.poly.transform(np.log1p(X_raw)))

        # 3. ResNet Prediction
        with torch.no_grad():
            tensor_in = torch.tensor(X_proc, dtype=torch.float32).to(self.device)
            p_res = torch.softmax(self.resnet(tensor_in), 1).cpu().numpy()

        # 4. Gradient Boosting Prediction
        p_gb = self.gb_model.predict_proba(X_proc)

        # 5. Ensemble (0.6 * ResNet + 0.4 * GB)
        final_probs = 0.6 * p_res + 0.4 * p_gb
        prediction = np.argmax(final_probs)

        return self.classes[prediction], final_probs[0]

# --- 3. LIVE HALO CONNECTION ---
def run_live_bci():
    predictor = LivePredictor()

    print("[Halo] Connecting to device...")
    acquisition.connect()

    with acquisition.Streaming() as stream:
        all_channels = stream.get_channel_names()
        fs = stream.get_sampling_rate()
        print(f"[Halo] Connected. Channels: {all_channels}, FS: {fs}Hz")

        # Map specific channels
        try:
            ch_indices = [all_channels.index(ch) for ch in TARGET_CHANNELS]
        except ValueError:
            print(f"[Error] Required channels {TARGET_CHANNELS} not found on device.")
            return

        buffer = np.zeros((0, 4))

        print("\n=== STARTING LIVE PREDICTION ===")
        print("Model: ResNet + GradientBoosting Ensemble")

        while True:
            chunk = stream.get_data()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue

            # Handle Transpose (We need Samples x Channels)
            if chunk.shape[0] == len(all_channels): chunk = chunk.T

            # Select channels
            chunk = chunk[:, ch_indices]

            # --- CRITICAL UNIT CHECK ---
            # Training data (GameEmo) is usually uV (values ~10-100)
            # BrainAccess is usually Volts (values ~0.00005)
            # We scale up if numbers are tiny to match training distribution
            if np.mean(np.abs(chunk)) < 0.001:
                chunk = chunk * 1e6

            buffer = np.vstack([buffer, chunk])

            # Process when we have 1 second of data
            if len(buffer) >= WINDOW_SIZE:
                window = buffer[-WINDOW_SIZE:]

                label, probs = predictor.predict(window)

                # Visual Output
                probs_str = " ".join([f"{p:.2f}" for p in probs])
                color = "\033[92m" if label in ['Funny', 'Calm'] else "\033[91m"
                print(f"\r{color}Pred: {label}\033[0m [ {probs_str} ]", end="")

                # Sliding Window: Keep 75% of old data for smoothness
                overlap = int(WINDOW_SIZE * 0.75)
                buffer = buffer[-overlap:]

            time.sleep(0.01)

if __name__ == "__main__":
    run_live_bci()
