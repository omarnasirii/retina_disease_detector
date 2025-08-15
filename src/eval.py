import argparse
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from src.data_loader import RetinalDataset, get_transforms
from src.inference import run_inference

def main(args):
    import pandas as pd
    df = pd.read_csv(args.csv)
    labels = df['diagnosis'].astype(int).tolist()
    preds = []
    for i, (_, row) in enumerate(df.iterrows()):
        image_path = f"{args.img_dir}/{row['id_code']}.png"
        result = run_inference(args.weights, image_path, args.device, return_cam=False)
        preds.append(result['class'])
        if i % 50 == 0:
            print(f"Processed {i+1}/{len(df)} images")
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    cm = confusion_matrix(labels, preds)
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on retinal dataset")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with image and label columns')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on')
    args = parser.parse_args()
    main(args)
