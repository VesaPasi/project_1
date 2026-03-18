## Manages the training and validation processes, including model initialization
## and execution of training epochs, ensuring optimal performance

import os
import torch
import torch.optim as optim
from args import get_args

def train_model(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(output['boxes'], target['boxes']) + 
                       criterion(output['labels'], target['labels']) 
                       for output, target in zip(outputs, targets))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_score = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss = sum(criterion(output['boxes'], target['boxes']) + 
                       criterion(output['labels'], target['labels']) 
                       for output, target in zip(outputs, targets))
            val_loss += loss.item()
            # Here you can calculate your validation score (e.g., mAP) and accumulate it

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_score = val_score / len(val_loader)  # Replace with actual score calculation

    return val_epoch_loss, val_epoch_score

