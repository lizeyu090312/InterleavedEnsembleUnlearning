import torch
import torch.nn as nn
from functools import partial

def wide_resnet50_2(num_classes, *args, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    model.fc = nn.Linear(512 * 4, num_classes)  # 512 * block.expansion, where block.expansion == 4
    return model

def resnet18(num_classes, *args, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.fc = nn.Linear(512 * 1, num_classes)  # 512 * block.expansion, where block.expansion == 1
    return model
