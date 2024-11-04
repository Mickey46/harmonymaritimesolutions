from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import base64
from PIL import Image
import io
import numpy as np
import cv2
from unittest.mock import patch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
import os
import random

app = FastAPI()

# Allow all origins for simplicity (you might want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static-live", StaticFiles(directory="static-live"), name="static-live")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

fake_users_db = {
    "user1": {
        "username": "user1",
        "full_name": "User One",
        "email": "user1@example.com",
        "hashed_password": "fakehashedpassword1",
        "disabled": False,
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict

def authenticate_user(fake_users_db, username: str, password: str):
    user = get_user(fake_users_db, username)
    if not user:
        return False
    if not user["hashed_password"] == fake_hash_password(password):
        return False
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password"
        )
    return {"access_token": user["username"], "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = get_user(fake_users_db, token)
    if not user:
        raise HTTPException(
            status_code=400, detail="Invalid authentication credentials"
        )
    return user

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("./static-live/index.html", "r") as f:
        return f.read()

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("./static-live/login.html", "r") as f:
        return f.read()

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for flash_attn import issues."""
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Apply the patch
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).cuda()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

def draw_bbox(image, data):
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = map(int, bbox)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            image_data = data['image'].split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            task = data['task']
            
            if task == 'concise-object-detection':
                result = run_example('<OD>', image)
                np_image = np.array(image)
                annotated_image = draw_bbox(np_image, result['<OD>'])
            elif task == 'detailed-object-detection':
                result = run_example('<DENSE_REGION_CAPTION>', image)
                np_image = np.array(image)
                annotated_image = draw_bbox(np_image, result['<DENSE_REGION_CAPTION>'])
            elif task in ['short-caption', 'medium-caption', 'long-caption']:
                caption_tasks = {
                    'short-caption': '<CAPTION>',
                    'medium-caption': '<DETAILED_CAPTION>',
                    'long-caption': '<MORE_DETAILED_CAPTION>'
                }
                result = run_example(caption_tasks[task], image)
                await websocket.send_json({"caption": result[caption_tasks[task]]})
                continue
            elif task == 'find-object':
                object_to_find = data['object']
                result = run_example('<CAPTION_TO_PHRASE_GROUNDING>', image, text_input=object_to_find)
                np_image = np.array(image)
                annotated_image = draw_bbox(np_image, result['<CAPTION_TO_PHRASE_GROUNDING>'])
            else:
                await websocket.send_json({"error": "Invalid task"})
                continue
            
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(buffer).decode()
            
            await websocket.send_json({"image": f"data:image/jpeg;base64,{img_str}"})
    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

