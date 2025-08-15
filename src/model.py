import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-based classifier for 5-class retinal disease grading.
    """
    def __init__(self, num_classes=5, backbone='efficientnet_b3', pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)