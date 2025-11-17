import os
import pandas as pd
import torch
from args import get_args
from dataset import Knee_Dataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model

def main():
    args = get_args()

    train_set  = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))   
    val_set    = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    # Create dataset instances
    train_dataset = Knee_Dataset(train_set)
    val_dataset = Knee_Dataset(val_set)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

   # Initialize model
    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained=False,
        path_pretrained="",
        restore_weights=False,
        path_weights=''
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader, val_loader, args)


if __name__ == "__main__":
    main()