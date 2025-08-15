import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import base64
import cv2
from torchvision import transforms
from src.model import EfficientNetClassifier
from src.data_loader import get_transforms

def load_model(weights_path, device):
    model = EfficientNetClassifier(num_classes=5)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = get_transforms(train=False)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
    return pred_class, probs

def generate_gradcam(model, image_tensor, device, target_class=None):
    # Grad-CAM for EfficientNet: use hooks on last conv layer
    last_conv = model.backbone.features[-1]
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = last_conv.register_forward_hook(forward_hook)
    handle_bwd = last_conv.register_full_backward_hook(backward_hook)

    model.eval()
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    score = output[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    # Get the activations and gradients
    act = activations[0].cpu().numpy()[0]
    grad = gradients[0].cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))
    cam = (weights[:, None, None] * act).sum(axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (300, 300))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(255 * cam)

    orig_img = image_tensor.squeeze().cpu().numpy().transpose(1,2,0)
    orig_img = (orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    orig_img = np.clip(orig_img, 0, 1)
    orig_img = np.uint8(255 * orig_img)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
    _, buffer = cv2.imencode('.png', overlay)
    b64 = base64.b64encode(buffer).decode('utf-8')

    handle_fwd.remove()
    handle_bwd.remove()
    return b64

def run_inference(weights_path, image_path, device='cpu', return_cam=True):
    model = load_model(weights_path, device)
    image_tensor = preprocess_image(image_path)
    pred_class, probs = predict(model, image_tensor, device)
    cam_b64 = generate_gradcam(model, image_tensor, device, pred_class) if return_cam else None
    return {
        'class': pred_class,
        'probabilities': probs.tolist(),
        'gradcam': cam_b64
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a retinal image")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on')
    parser.add_argument('--no-cam', action='store_true', help='Do not generate Grad-CAM overlay')
    args = parser.parse_args()
    result = run_inference(args.weights, args.image, args.device, not args.no_cam)
    print(result)
