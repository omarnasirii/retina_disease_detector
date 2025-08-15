from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import torch
from src.inference import run_inference
import os

app = FastAPI()

MODEL_WEIGHTS = os.getenv('MODEL_WEIGHTS', 'checkpoints/best_model_fold1.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.post('/predict/')
def predict(image: UploadFile = File(...), return_cam: bool = Form(True)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(image.file.read())
        tmp_path = tmp.name
    result = run_inference(MODEL_WEIGHTS, tmp_path, DEVICE, return_cam)
    os.remove(tmp_path)
    response = {
        'class': result['class'],
        'probabilities': result['probabilities'],
    }
    if return_cam and result['gradcam']:
        response['gradcam'] = result['gradcam']
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deployment.app:app", host="0.0.0.0", port=8000, reload=True)
