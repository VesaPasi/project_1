##  Manages data loaders and starts the training.

# import args from python file args.py in the root directory
from args import get_args
# import dataset
from dataset import ObjDetectionDataset
# import dataloader
from torch.utils.data import DataLoader

# import model
from model import build_model

from pathlib import Path

import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    axes[0].plot(metrics['epoch'], metrics['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(metrics['epoch'], metrics['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation score
    axes[1].plot(metrics['epoch'], metrics['val_score'], label='Val Score', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Detection Accuracy (%)')
    axes[1].set_title('Validation Detection Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'training_metrics.png'")
    plt.show()

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()

    # 1. Read the dataframes
    train_csv = Path(args.csv_dir) / 'train_data.csv'
    val_csv = Path(args.csv_dir) / 'val_data.csv'
    print(f"Train CSV path: {train_csv}")
    print(f"Validation CSV path: {val_csv}")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # 2. Prepare datasets
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    images, targets = next(iter(train_loader))
    print(f"Batch of images: {len(images)}")
    print(f"Batch of targets: {len(targets)}")

    # 4. Initiate the model
    model = build_model(args.backbone)

    # 5. Start training
    from trainer import train_model
    metrics = train_model(model, train_loader, val_loader, args)
    
    # 6. Plot metrics
    plot_metrics(metrics)

    
    
if __name__ == '__main__':
    main()