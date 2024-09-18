from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
from PIL import Image
import io
import numpy as np
import cv2
import random
from ultralytics import YOLO

app = FastAPI()

# Allow all origins for simplicity (you might want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, index.html, etc.) from the 'static-live' directory
app.mount("/static-live", StaticFiles(directory="static-live"), name="static-live")

# Serve the index.html page from the 'static-live' directory when accessing the root URL
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(open("static-live/index.html").read())

# Load the YOLOv8 model for zero-shot object detection
model = YOLO("yolov8n.pt")  # You can replace this with any YOLOv8 model (e.g., 'yolov8s.pt')

def run_yolov8(image):
    # Perform detection using YOLOv8
    results = model(image)
    return results[0]  # Return first result (since YOLOv8 can handle batches)

def draw_bbox(image, results, object_to_find=None):
    # Use `model.names` to convert label indices to class names
    for bbox, label_idx, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = model.names[int(label_idx)]  # Get the label name from the index
        
        if object_to_find and object_to_find.lower() in label.lower():  # Filter based on the search query
            x1, y1, x2, y2 = map(int, bbox)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        elif not object_to_find:
            x1, y1, x2, y2 = map(int, bbox)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    try:
        while True:
            try:
                data = await websocket.receive_json()
                print(f"Data received: {data}")
                
                image_data = data['image'].split(',')[1]
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                task = data['task']
                object_to_find = data.get('object', None)  # Get object to find from the user input

                np_image = np.array(image)
                annotated_image = None

                # Manual Detection: Find specified object only
                if task == 'manual':
                    if object_to_find:
                        print(f"Object to find: {object_to_find}")
                        result = run_yolov8(np_image)
                        annotated_image = draw_bbox(np_image, result, object_to_find)
                    else:
                        annotated_image = np_image  # No annotation if no object specified
                # Automatic Detection: Detect all objects
                elif task == 'automatic':
                    result = run_yolov8(np_image)
                    annotated_image = draw_bbox(np_image, result)

                # Convert back to base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(buffer).decode()

                print("Sending annotated image back")
                await websocket.send_json({"image": f"data:image/jpeg;base64,{img_str}"})
            except Exception as e:
                print(f"Error processing data: {e}")
                await websocket.send_json({"error": str(e)})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
