"""resnet18_scratch: ImageNet ResNet-18 trained from scratch on the six base classes (embedding = pooled 512-d)."""
import os
import torch
from torch import nn
from torchvision import models

WEIGHTS = os.environ.get('EUROSAT_RESNET18_SCRATCH_WEIGHTS',
                         '/content/drive/MyDrive/EuroSATRepresentationBenchmark/cache/weights/resnet18_scratch.pth')


def build():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
    model.fc = nn.Identity()
    return model
