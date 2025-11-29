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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_sent_data = None
    send_count = 0
    try:
        while True:
            data = detector.get_data()
            send_count += 1
            
            # Check if data actually changed
            data_hash = hash(json.dumps(data))
            data_changed = (data_hash != last_sent_data) if last_sent_data is not None else True
            last_sent_data = data_hash
            
            # SOLID FIX: Enhanced WebSocket logging with data freshness check
            probs = data.get('probabilities', [])
            prob_str = ', '.join([f'{p:.4f}' for p in probs]) if isinstance(probs, list) else str(probs)
            data_age = data.get('data_age', 0)
            buffer_fill_rate = data.get('buffer_fill_rate', 0)
            prediction_rate = data.get('prediction_rate', 0)
            
            # Log every 10th message to avoid spam, but always log warnings
            should_log = (send_count % 10 == 0) or not data_changed or data_age > 0.5
            
            if should_log:
                print(f"[WEBSOCKET #{send_count}] Sending data | "
                      f"Changed: {data_changed} | "
                      f"Age: {data_age:.3f}s | "
                      f"Fill rate: {buffer_fill_rate:.1f} Hz | "
                      f"Pred rate: {prediction_rate:.1f}/s | "
                      f"Emotion: {data.get('emotion', 'N/A')} | "
                      f"Probs: [{prob_str}]")
            
            # Warn if data is stale or not changing
            if data_age > 1.0:
                print(f"[WEBSOCKET WARNING #{send_count}] ⚠ Data is STALE ({data_age:.2f}s old)! Predictions may not be running!")
            elif not data_changed and send_count > 10:
                print(f"[WEBSOCKET WARNING #{send_count}] ⚠ Sending same data again! Probabilities are not updating!")
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1) # 10Hz update rate
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
