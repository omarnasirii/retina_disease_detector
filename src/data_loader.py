import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RetinalDataset(Dataset):
    """
    PyTorch Dataset for retinal images and severity labels.
    Loads image paths and labels from a CSV file.
    Applies transforms for augmentation and normalization.
    """
    def __init__(self, data, img_dir, transform=None):
        # Accept either a DataFrame or a CSV file path
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # For APTOS 2019: image filename is id_code + '.png', label is diagnosis
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx]['id_code']) + '.png')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx]['diagnosis'])
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(train=True):
    """
    Returns torchvision transforms for training or validation.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])