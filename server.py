from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
from brainaccess_live import EmotionDetector

app = FastAPI()

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
    try:
        while True:
            data = detector.get_data()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.1) # 10Hz update rate
    except Exception as e:
        print(f"WebSocket error: {e}")
