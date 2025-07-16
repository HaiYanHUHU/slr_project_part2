import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3Extractor(nn.Module):
    def __init__(self):
        super(MobileNetV3Extractor, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.features(x)
            x = self.pool(x)
            x = x.view(B, T, -1)
            return x
        else:
            # 
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x

