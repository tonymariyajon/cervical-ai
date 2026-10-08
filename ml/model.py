"""
Baseline CNN Architecture for 6-Class Cervical Cell Classification.

A clean, modular convolutional neural network designed for 224x224 RGB cell patches.
Outputs unnormalized logits for the 6 Bethesda diagnostic classes.
"""

import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """Simple 4-stage convolutional neural network for multi-class cell classification.

    Input: (B, 3, 224, 224)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor: 4 convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling: reduces 14x14 feature map to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

def get_model(num_classes: int = 6) -> BaselineCNN:
    """Instantiate baseline CNN model."""
    return BaselineCNN(num_classes=num_classes)
