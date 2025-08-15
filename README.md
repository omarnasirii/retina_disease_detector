# Retinal Disease Severity Grading

A full PyTorch-based pipeline for training, evaluating, and deploying a retinal disease severity classifier using retinal fundus images (APTOS 2019 format). Includes model training, evaluation, Grad-CAM visualization, and a Gradio web demo.

---

## Features
- EfficientNet-B3 backbone for classification
- 5-class severity grading (No Disease, Mild, Moderate, Severe, Proliferative)
- Stratified K-Fold cross-validation
- Grad-CAM visualization for model interpretability
- Gradio web demo for easy inference and visualization
- Modular code: training, evaluation, inference, and utilities

---

## Project Structure
```
├── app.py                  # Gradio demo entry point
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── model.py            # Model definition (EfficientNet)
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   ├── inference.py        # Inference and Grad-CAM
│   └── utils.py            # Utility functions
├── deployment/
│   ├── gradio_demo.py      # (Alternative) Gradio demo
│   └── requirements.txt    # (Optional) deployment dependencies
├── data/                   # Place your data here (not included)
│   ├── train_images/       # Training images
│   └── train.csv           # CSV with image ids and labels
├── checkpoints/            # Model weights (not included)
└── ...
```

---

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. **Create a virtual environment and install dependencies:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Download the APTOS 2019 dataset** from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).
2. **Organize your data:**
   - Place all training images in `data/train_images/`
   - Place the CSV file as `data/train.csv`

---

## Training

Train the model from scratch (will save weights to `checkpoints/`):
```sh
python src/train.py --data_csv data/train.csv --image_dir data/train_images --output_dir checkpoints/
```
- You can adjust arguments (epochs, batch size, etc.) in `src/train.py` or via command line if supported.

---

## Evaluation

Evaluate a trained model:
```sh
python src/eval.py --data_csv data/val.csv --image_dir data/val_images --weights checkpoints/best_model_fold1.pth
```
- Replace with your validation CSV and image directory as needed.

---

## Inference & Grad-CAM Visualization

Run inference and get Grad-CAM overlays:
```sh
python src/inference.py --image_path <path_to_image> --weights checkpoints/best_model_fold1.pth
```

---

## Gradio Web Demo

Launch the interactive web demo:
```sh
python app.py
```
- Open the link shown in your terminal (usually http://127.0.0.1:7860)
- Upload a retinal image to get predictions and Grad-CAM visualization

---

## Model Weights
- **Model weights are NOT included in this repo due to file size limits.**
- You can train your own using the instructions above.
- (Optional) If you want to use pre-trained weights, download them from a public link (Google Drive, Hugging Face Hub, etc.) and place them in the `checkpoints/` folder.

---

## Notes
- This project is for educational and research purposes only. Not for clinical use.
- For any issues, please open an issue or pull request on GitHub.

---

## License
[MIT License](LICENSE)
