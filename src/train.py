import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from src.data_loader import RetinalDataset, get_transforms
from src.model import EfficientNetClassifier
from src.utils import seed_everything


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}")
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import pandas as pd
    df = pd.read_csv(args.csv)
    labels = df['diagnosis'].astype(int).tolist()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    best_acc = 0.0
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"Fold {fold+1}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        train_dataset = RetinalDataset(train_df, args.img_dir, transform=get_transforms(train=True))
        val_dataset = RetinalDataset(val_df, args.img_dir, transform=get_transforms(train=False))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = EfficientNetClassifier(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        best_fold_acc = 0.0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save(model.state_dict(), f"{args.output_dir}/best_model_fold{fold+1}.pth")
        best_acc = max(best_acc, best_fold_acc)
    print(f"Best validation accuracy across folds: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet for Retinal Disease Severity Grading")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with image and label columns')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
