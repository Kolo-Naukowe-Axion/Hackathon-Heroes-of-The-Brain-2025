# Neurohackathon - EEG-Based Emotion Detection & Music Therapy

A real-time brain-computer interface (BCI) system that reads EEG brainwave data, classifies emotions using a weighted ensemble of deep learning and gradient boosting models, and recommends music playlists to help users achieve a neutral emotional state. Designed for pre-operative and therapeutic applications.

## 🧠 Overview

This project combines:
- **EEG Data Acquisition**: Real-time brainwave data from BrainAccess EEG headset via LSL (Lab Streaming Layer)
- **Emotion Classification**: Ensemble model combining:
  - WideResNet (Deep Learning) - 60% weight
  - Gradient Boosting Classifier - 40% weight
- **Music Therapy**: Spotify integration to recommend curated playlists based on detected emotional state
- **3D Visualization**: Interactive brain visualization with emotion-based color mapping

## 🏗️ Architecture

```
┌─────────────────┐
│  BrainAccess    │
│  EEG Headset    │
└────────┬────────┘
         │ LSL Stream
         ▼
┌─────────────────┐
│  Backend        │
│  (FastAPI)      │
│  - Emotion      │
│    Detection    │
│  - WebSocket    │
└────────┬────────┘
         │ WebSocket
         ▼
┌─────────────────┐
│  Frontend       │
│  (React + 3D)   │
│  - Visualization│
│  - Spotify      │
│    Integration  │
└─────────────────┘
```

## 📋 Prerequisites

### Hardware
- BrainAccess EEG headset (or compatible LSL-compatible EEG device)
- BrainAccess Board application running with LSL streaming enabled

### Software
- Python 3.8+
- Node.js 18+
- Spotify Developer Account (for music playback)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Neurohackathon
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

The backend will automatically load the pre-trained models from `model_v5/`:
- `resnet_weights.pth` - WideResNet model weights
- `ml_components.pkl` - Scaler, polynomial features, and gradient boosting model

### 3. Frontend Setup

```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend` directory:

```bash
cp .env.example .env
```

Edit `.env` and add your Spotify Client ID:
```
VITE_SPOTIFY_CLIENT_ID=your_spotify_client_id_here
VITE_SPOTIFY_REDIRECT_URI=http://127.0.0.1:5173/
```

**Getting a Spotify Client ID:**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Copy the Client ID to your `.env` file
4. Add `http://127.0.0.1:5173/` as a redirect URI in your app settings

### 4. Start the Application

**Terminal 1 - Backend:**
```bash
cd backend
python server.py
```

The backend will:
- Search for LSL EEG streams
- Connect to BrainAccess device
- Start emotion detection pipeline
- Serve WebSocket at `ws://localhost:8000/ws`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://127.0.0.1:5173/`

### 5. Connect EEG Device

1. Ensure BrainAccess Board is running
2. Connect your EEG headset
3. Enable LSL streaming in BrainAccess Board:
   - Go to **Stream** or **LSL** tab
   - Enable **LSL Stream**
   - Ensure stream type is `EEG`

The backend will automatically detect and connect to the stream.

## 🎯 How It Works

### Emotion Detection Pipeline

1. **EEG Data Acquisition**: Receives 125Hz EEG data from 4 channels (AF3, AF4, O1, O2)
2. **Feature Extraction**: Computes relative band powers for 5 frequency bands:
   - Delta (0.5-4 Hz)
   - Theta (4-8 Hz)
   - Alpha (8-13 Hz)
   - Beta (13-30 Hz)
   - Gamma (30-45 Hz)
3. **Preprocessing**: 
   - Log transformation (`log1p`)
   - Polynomial feature expansion
   - Standard scaling
4. **Ensemble Prediction**:
   - WideResNet (60% weight) - Deep learning model
   - Gradient Boosting (40% weight) - Traditional ML model
   - Weighted average of probabilities
5. **Smoothing**: Temporal smoothing over last 10 predictions (~1 second)
6. **Emotion Mapping**: Maps model outputs to frontend emotions:
   - `Boring` → `neutral`
   - `Calm` → `calm`
   - `Horror` → `angry`
   - `Funny` → `happy`

### Music Recommendation

Based on detected emotion, the system recommends curated Spotify playlists designed to guide users toward a neutral state:

- **Neutral**: Maintenance playlists
- **Calm**: Energizing playlists
- **Happy**: Calming playlists
- **Sad**: Uplifting playlists
- **Angry**: Soothing playlists

## 📁 Project Structure

```
Neurohackathon/
├── backend/
│   ├── server.py              # FastAPI server with WebSocket
│   ├── brainaccess_live.py    # EEG processing & emotion detection
│   ├── model_v5/              # Pre-trained models
│   │   ├── resnet_weights.pth
│   │   ├── ml_components.pkl
│   │   └── live_bci_runner.py
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main application
│   │   ├── components/       # React components
│   │   │   ├── BrainHero.jsx # 3D brain visualization
│   │   │   ├── BackgroundParticles.jsx
│   │   │   └── DebugPanel.jsx
│   │   └── utils/            # Spotify & auth utilities
│   ├── public/
│   │   └── models/           # 3D brain models
│   ├── .env.example          # Environment template
│   └── package.json
└── README.md                  # This file
```

## 🔧 Configuration

### Backend Settings

Edit `backend/brainaccess_live.py` to adjust:

- `WINDOW_SIZE`: EEG window size (default: 250 samples = 1 second)
- `PREDICTION_INTERVAL`: Prediction frequency (default: 0.1s = 10Hz)
- `SMOOTHING_WINDOW`: Number of predictions to average (default: 10)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for emotion detection (default: 0.35)

### Frontend Settings

- Debug mode: Press `D` key to toggle debug panel
- Manual emotion testing: Press `N` (Neutral), `C` (Calm), `H` (Happy), `S` (Sad), `A` (Angry)

## 🐛 Troubleshooting

### Backend Issues

**"No EEG stream found"**
- Ensure BrainAccess Board is running
- Verify LSL streaming is enabled
- Check firewall settings (LSL uses UDP broadcast)
- Try restarting BrainAccess Board

**"Model loading error"**
- Verify `model_v5/resnet_weights.pth` exists
- Verify `model_v5/ml_components.pkl` exists
- Check file permissions

**"Connection timeout"**
- Ensure EEG device is connected to BrainAccess Board
- Check device battery/power
- Verify USB/Bluetooth connection

### Frontend Issues

**"Spotify authentication failed"**
- Verify `.env` file has correct `VITE_SPOTIFY_CLIENT_ID`
- Check redirect URI matches Spotify app settings
- Clear browser cache and localStorage

**"WebSocket connection failed"**
- Ensure backend server is running on port 8000
- Check browser console for connection errors
- Verify CORS settings in `server.py`

## 📊 Model Details

### Architecture

- **WideResNet**: 1024-dim hidden layers, 3 residual blocks, dropout 0.3-0.4
- **Gradient Boosting**: Scikit-learn implementation with optimized hyperparameters
- **Input Features**: 20 features (4 channels × 5 bands) → expanded via polynomial features
- **Output Classes**: 4 emotions (Boring, Calm, Horror, Funny)

### Training Data

The models were trained on EEG data collected during emotional state induction, with labels validated through self-reporting and physiological markers.

## 🤝 Contributing

This project was developed for a neurohackathon. Contributions are welcome!

## 🙏 Acknowledgments

- BrainAccess for EEG hardware and LSL integration
- Spotify for music API
- React Three Fiber for 3D visualization
- FastAPI for backend framework

## 📧 Contact

iwo.smura@gmail.com

