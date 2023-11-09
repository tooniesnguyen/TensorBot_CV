import uvicorn
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np



app = FastAPI()





@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    vid = cv2.VideoCapture(0)  # Khởi tạo camera ở đây
    while True:
        ret, frame = vid.read()
        await websocket.send_bytes(frame)




if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)