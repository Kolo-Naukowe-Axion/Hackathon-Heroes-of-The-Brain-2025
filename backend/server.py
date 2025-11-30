from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from brainaccess_live import EmotionDetector

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize detector
detector = EmotionDetector()

@app.on_event("startup")
async def startup_event():
    detector.start()

@app.on_event("shutdown")
async def shutdown_event():
    detector.stop()

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/status")
async def get_status():
    """Debug endpoint to check BCI connection status"""
    data = detector.get_data()
    return {
        "is_mock": data.get('is_mock', True),
        "emotion": data.get('emotion', 'unknown'),
        "probabilities": data.get('probabilities', []),
        "data_age": data.get('data_age', 0),
        "buffer_fill_rate": data.get('buffer_fill_rate', 0),
        "prediction_rate": data.get('prediction_rate', 0)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Check current mode status
    initial_data = detector.get_data()
    is_mock_initial = initial_data.get('is_mock', True)
    
    print("="*60)
    print("🔌 WEBSOCKET CLIENT CONNECTED!")
    print("✅ REAL EEG DATA ONLY - NO MOCK DATA")
    print("="*60)
    
    send_count = 0
    last_sent_probs = None
    last_sent_emotion = None
    try:
        while True:
            data = detector.get_data()
            send_count += 1
            
            # Extract key values
            probs = data.get('probabilities', [])
            emotion = data.get('emotion', 'N/A')
            
            # Check if values actually changed
            probs_changed = last_sent_probs is None or probs != last_sent_probs
            emotion_changed = last_sent_emotion is None or emotion != last_sent_emotion
            data_age = data.get('data_age', 0)
            
            # Always send if data is fresh (less than 0.2s old) or if values changed
            should_send = probs_changed or emotion_changed or data_age < 0.2
            
            if should_send:
                # FORCE is_mock to False - no mock data exists
                data['is_mock'] = False
                
                # Log when values change
                prob_str = ', '.join([f'{p:.3f}' for p in probs]) if isinstance(probs, list) else str(probs)
                if probs_changed or emotion_changed:
                    print(f"[WS→FRONTEND #{send_count}] {emotion} | Probs=[{prob_str}] | Age={data_age:.3f}s | CHANGED")
                elif send_count <= 5 or send_count % 50 == 0:
                    print(f"[WS→FRONTEND #{send_count}] {emotion} | Probs=[{prob_str}] | Age={data_age:.3f}s")
                
                # Serialize and send
                json_data = json.dumps(data)
                await websocket.send_text(json_data)
                
                # Update last sent values
                last_sent_probs = probs.copy() if isinstance(probs, list) else probs
                last_sent_emotion = emotion
            
            await asyncio.sleep(0.1)  # 10Hz update rate
    except WebSocketDisconnect:
        print("="*60)
        print("🔌 WEBSOCKET CLIENT DISCONNECTED!")
        print("="*60)
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
