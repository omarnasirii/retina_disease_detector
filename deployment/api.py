from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.inference import run_inference
from PIL import Image
import io
import os

app = FastAPI()

MODEL_WEIGHTS = os.getenv('MODEL_WEIGHTS', 'checkpoints/best_model_fold1.pth')
DEVICE = 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu'

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        temp_path = "temp.png"
        with open(temp_path, "wb") as f:
            f.write(output.getvalue())
        result = run_inference(MODEL_WEIGHTS, temp_path, DEVICE, return_cam=True)
        os.remove(temp_path)
    return JSONResponse({
        "class": int(result["class"]),
        "probabilities": [float(x) for x in result["probabilities"]],
        "gradcam": result["gradcam"]
    })
