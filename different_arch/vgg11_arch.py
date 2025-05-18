import torch
import torch.nn as nn
from functools import partial

def vgg(num_classes, *args, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    return model

