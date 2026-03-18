## Defines the neural network architecture

import torch
import torchvision
import args
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(backbone: str):
    '''Builds a Faster R-CNN model with the specified backbone architecture.'''
    if backbone == "fasterrcnn_resnet50_fpn":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, weights=weights)

    elif backbone == "fasterrcnn_mobilenet_V3":
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, weights=weights)

    return model