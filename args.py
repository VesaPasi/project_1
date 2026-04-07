## Defines arguments for refining the pipeline 

import argparse
import os
from pathlib import Path

import torch

csv_dir = Path("F:/data/CSVs")
out_dir = Path("F:/sessions")

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    # model choices
    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet50_fpn', 
                        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_V3"])
    # Dataset CSV directory
    parser.add_argument('--csv_dir', type=str, default=str(csv_dir))
    # Saved best model directory
    parser.add_argument('--out_dir', type=str, default=str(out_dir))
    # Input image size for train/val transforms
    parser.add_argument('--image_size', type=int, default=512)
    # Batch size
    parser.add_argument('--batch_size', type=int, default=8, choices=[8, 16, 32, 64])
    # Epochs
    parser.add_argument('--epochs', type=int, default=100)
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.001)
    # Weight decay
    parser.add_argument('--wd', type=float, default=1e-4)

    return parser.parse_args()