## Manages the training and validation processes, including model initialization
## and execution of training epochs, ensuring optimal performance

import os
import torch
import torch.optim as optim
from torchvision.ops import box_iou
from args import get_args

def compute_iou_loss(predictions, targets):
    """
    Compute IoU-based loss for object detection.
    Returns average loss across all predictions in a batch.
    """
    total_loss = 0.0
    num_predictions = 0
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
            continue
            
        # Compute IoU between predicted and ground truth boxes
        ious = box_iou(pred['boxes'], target['boxes'])
        
        # Use 1 - max_iou as loss (higher IoU = lower loss)
        max_ious = ious.max(dim=1)[0]
        loss = (1.0 - max_ious).sum()
        
        total_loss += loss.item()
        num_predictions += len(pred['boxes'])
    
    return total_loss / max(num_predictions, 1)

def compute_detection_score(predictions, targets):
    """
    Compute detection accuracy based on IoU threshold (0.5).
    Returns percentage of predictions with IoU >= 0.5.
    """
    total_correct = 0
    total_predictions = 0
    iou_threshold = 0.5
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
            continue
            
        ious = box_iou(pred['boxes'], target['boxes'])
        max_ious = ious.max(dim=1)[0]
        
        # Count predictions where max IoU >= threshold
        correct = (max_ious >= iou_threshold).sum().item()
        total_correct += correct
        total_predictions += len(pred['boxes'])
    
    return (total_correct / max(total_predictions, 1)) * 100  # Return as percentage

def train_model(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Store metrics for plotting
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_score': []
    }

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            # In training mode, torchvision detection models require targets
            # and return a dict of detection losses.
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_score = validate_model(model, val_loader, device)
        
        # Store metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_score'].append(val_score)
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f}")
    
    return metrics

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            
            # Calculate loss based on IoU
            val_loss += compute_iou_loss(predictions, targets)
            
            # Calculate detection score (% of predictions with IoU >= 0.5)
            val_score += compute_detection_score(predictions, targets)

    val_epoch_loss = val_loss / max(len(val_loader), 1)
    val_epoch_score = val_score / max(len(val_loader), 1)

    return val_epoch_loss, val_epoch_score

