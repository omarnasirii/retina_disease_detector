import gradio as gr
import torch
from src.inference import run_inference
import os
from PIL import Image
import io
import base64

MODEL_WEIGHTS = os.getenv('MODEL_WEIGHTS', 'checkpoints/best_model_fold1.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSES = ['No Disease', 'Mild', 'Moderate', 'Severe', 'Proliferative']

def predict_gradio(image):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        image.save(tmp.name)
        result = run_inference(MODEL_WEIGHTS, tmp.name, DEVICE, return_cam=True)
        os.remove(tmp.name)
    class_idx = result['class']
    probs = result['probabilities']
    gradcam_b64 = result['gradcam']
    gradcam_img = None
    if gradcam_b64:
        gradcam_img = Image.open(io.BytesIO(base64.b64decode(gradcam_b64)))
    return (
        CLASSES[class_idx],
        {c: float(p) for c, p in zip(CLASSES, probs)},
        gradcam_img
    )

demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil", label="Upload Retinal Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicted Class"),
        gr.JSON(label="Probabilities"),
        gr.Image(type="pil", label="Grad-CAM Overlay")
    ],
    title="Retinal Disease Severity Grading",
    description="Upload a retinal image to get severity prediction and Grad-CAM visualization."
)

if __name__ == "__main__":
    demo.launch(share=True)
