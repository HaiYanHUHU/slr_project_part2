import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Extractor(nn.Module):
    def __init__(self):
        super(MobileNetV3Extractor, self).__init__()
        model = mobilenet_v3_large(pretrained=True)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [B, T, C, H, W] — Video frame sequence

        Returns:
            Tensor, shape [B, T, D] — Feature sequence of each frame
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.pool(x)              # [B*T, D, 1, 1]
        x = x.view(B, T, -1)          # [B, T, D]
        return x
