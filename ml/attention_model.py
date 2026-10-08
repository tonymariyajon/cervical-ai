"""
Attention-Enhanced CNN Architecture for 6-Class Cervical Cell Classification.

Extends the BaselineCNN with a lightweight Spatial Attention Mechanism (CBAM Spatial Attention).
The spatial attention module produces an attention map showing which spatial regions
(e.g., cell nucleus, chromatin boundaries, cytoplasm) receive higher importance.

Input: (B, 3, 224, 224)
Output: (B, num_classes) logits, and optionally (B, 1, 14, 14) spatial attention map.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM-style).

    Compresses channel information via average pooling and max pooling along the
    channel axis, then applies a spatial convolution followed by a sigmoid gate.
    Produces a 2D spatial weight map in range [0, 1] highlighting informative regions.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size should be 3 or 7"
        padding = kernel_size // 2
        # Takes 2 channels (avg-pool and max-pool across channels) and outputs 1 spatial mask
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial attention map and apply it element-wise.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            out: Attention-weighted feature map of shape (B, C, H, W).
            attention_map: Spatial attention weights of shape (B, 1, H, W).
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)

        # Generate spatial attention map
        attention_map = self.sigmoid(self.conv(combined))  # Shape: (B, 1, H, W)

        # Modulate features by attention map (broadcast over C channels)
        out = x * attention_map
        return out, attention_map

class AttentionCNN(nn.Module):
    """Convolutional Neural Network with Spatial Attention for cervical cell classification.

    Preserves the 4-stage convolutional backbone of BaselineCNN and inserts
    a Spatial Attention module before global pooling.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor blocks (identical to BaselineCNN)
        # Block 1: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Spatial Attention Module acting on the 14x14 feature representation
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input images of shape (B, 3, 224, 224).
            return_attention: If True, returns (logits, attention_map).
                              If False, returns logits only (standard training mode).
        """
        # Feature extraction
        x = self.block1(x)  # (B, 32, 112, 112)
        x = self.block2(x)  # (B, 64, 56, 56)
        x = self.block3(x)  # (B, 128, 28, 28)
        x = self.block4(x)  # (B, 256, 14, 14)

        # Apply Spatial Attention
        x_attended, attention_map = self.spatial_attention(x)  # (B, 256, 14, 14), (B, 1, 14, 14)

        # Pool and classify
        pooled = self.pool(x_attended)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attention_map
        return logits

def get_attention_model(num_classes: int = 6) -> AttentionCNN:
    """Instantiate the AttentionCNN model."""
    return AttentionCNN(num_classes=num_classes)
