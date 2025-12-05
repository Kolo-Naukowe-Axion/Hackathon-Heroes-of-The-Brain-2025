# Backend - EEG Emotion Detection Server

FastAPI backend server that processes real-time EEG data from BrainAccess devices and classifies emotions using an ensemble of deep learning and gradient boosting models.

## Features

- Real-time EEG data acquisition via LSL (Lab Streaming Layer)
- Emotion classification using weighted ensemble:
  - WideResNet (Deep Learning) - 60% weight
  - Gradient Boosting Classifier - 40% weight
- WebSocket API for real-time emotion updates
- Automatic reconnection to EEG streams
- Prediction smoothing for stable outputs

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure the following files exist in `model_v5/`:
- `resnet_weights.pth` - Pre-trained WideResNet model
- `ml_components.pkl` - Scaler, polynomial transformer, and gradient boosting model

### 3. Start the Server

```bash
python server.py
```

The server will:
- Start on `http://localhost:8000`
- WebSocket endpoint: `ws://localhost:8000/ws`
- Status endpoint: `http://localhost:8000/status`

## EEG Device Connection

### Using BrainAccess

1. **Start BrainAccess Board** application
2. **Connect your EEG headset** to the device
3. **Enable LSL Streaming**:
   - Go to **Stream** or **LSL** tab in BrainAccess Board
   - Enable **LSL Stream** toggle
   - Ensure stream type is set to `EEG`

The backend will automatically:
- Search for available LSL streams
- Connect to the EEG stream
- Start processing brainwave data
- Begin emotion predictions

### Expected EEG Stream

- **Channels**: 4 channels (AF3, AF4, O1, O2)
- **Sampling Rate**: 250 Hz
- **Stream Type**: `EEG`

## API Endpoints

### WebSocket: `/ws`

Real-time emotion updates. Messages are sent as JSON:

```json
{
  "emotion": "neutral",
  "probabilities": [0.25, 0.30, 0.20, 0.25],
  "is_mock": false,
  "data_age": 0.05,
  "buffer_fill_rate": 250.0,
  "prediction_rate": 10.0
}
```

**Fields:**
- `emotion`: Detected emotion label (neutral, calm, happy, sad, angry)
- `probabilities`: Class probabilities [Boring, Calm, Horror, Funny]
- `is_mock`: Always `false` (no mock data)
- `data_age`: Seconds since last update
- `buffer_fill_rate`: EEG sample rate (Hz)
- `prediction_rate`: Emotion prediction rate (Hz)

### GET `/status`

Debug endpoint to check connection status:

```bash
curl http://localhost:8000/status
```

Returns current detector state and connection information.

## Configuration

Edit `brainaccess_live.py` to adjust:

### Processing Parameters

```python
WINDOW_SIZE = 250         # EEG window size (1 second at 250Hz)
BUFFER_LENGTH = 250      # Buffer size
TARGET_FS = 250          # Target sampling frequency
PREDICTION_INTERVAL = 0.1  # Prediction frequency (10Hz)
SMOOTHING_WINDOW = 10    # Number of predictions to average
CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence for detection
```

### Model Configuration

```python
MODEL_TO_FRONTEND_MAP = {
    'Boring': 'neutral',
    'Calm': 'calm',
    'Horror': 'angry',
    'Funny': 'happy'
}
```

## Troubleshooting

### "No EEG stream found"

**Solutions:**
1. Verify BrainAccess Board is running
2. Check LSL streaming is enabled
3. Ensure device is connected and powered
4. Check firewall settings (LSL uses UDP broadcast)
5. Try restarting BrainAccess Board

### "Model loading error"

**Solutions:**
1. Verify `model_v5/resnet_weights.pth` exists
2. Verify `model_v5/ml_components.pkl` exists
3. Check file permissions
4. Ensure PyTorch is installed correctly

### "Connection timeout" or "No data received"

**Solutions:**
1. Check EEG device connection
2. Verify device battery/power
3. Restart BrainAccess Board
4. Check LSL stream is active in BrainAccess Board

### Low prediction rate

**Possible causes:**
- Insufficient EEG data (buffer not full)
- Device disconnection
- Network issues with LSL

Check the status endpoint or console logs for detailed diagnostics.

## Architecture

### Emotion Detection Pipeline

1. **LSL Stream Acquisition**: Receives EEG data chunks
2. **Buffer Management**: Maintains 1-second rolling window
3. **Feature Extraction**: Computes relative band powers (5 bands × 4 channels = 20 features)
4. **Preprocessing**:
   - Log transformation (`log1p`)
   - Polynomial feature expansion
   - Standard scaling
5. **Ensemble Prediction**:
   - WideResNet forward pass → probabilities
   - Gradient Boosting prediction → probabilities
   - Weighted average (0.6 × ResNet + 0.4 × GB)
6. **Temporal Smoothing**: Average over last N predictions
7. **Emotion Mapping**: Map model classes to frontend labels

### Model Details

- **WideResNet**: 1024-dim hidden, 3 residual blocks, dropout 0.3-0.4
- **Input**: 20 features → polynomial expansion → scaled
- **Output**: 4-class probabilities
- **Ensemble**: 60% ResNet + 40% Gradient Boosting

## Development

### Running Tests

```bash
# Test LSL connection
python -c "from pylsl import resolve_streams; print(resolve_streams())"

# Test model loading
python -c "from brainaccess_live import EmotionDetector; d = EmotionDetector(); print('Models loaded:', d.resnet is not None)"
```

### Standalone Mode

Run the detector directly:

```bash
python brainaccess_live.py
```

This will start the detector and print emotion predictions to console.

## Dependencies

Key dependencies:
- `fastapi` - Web framework
- `pylsl` - LSL stream handling
- `torch` - Deep learning (WideResNet)
- `scikit-learn` - Gradient boosting and preprocessing
- `numpy`, `scipy` - Signal processing
- `joblib` - Model serialization

See `requirements.txt` for complete list.
