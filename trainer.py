from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import dice_loss_from_logits
from utils import dice_score_from_logits

def train_model(model, train_loader, val_loader, args):
    args = get_args()
    model = model.to(args.device)

    criterion = nn.BCWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for data_batch in train_loader:
        images, masks = data_batch['image'].to(args.device), data_batch['mask'].to(args.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_bce = criterion(outputs, masks)
        

        loss_dice = dice_loss_from_logits(outputs, masks)
        loss = loss_bce + loss_dice
 
        loss.backward() 
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    val_loss, val_score = validate_model(model, val_loader, criterion, args)
    print(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_score:.4f}")

def validate_model(model, val_loader, loss_fn, args):
    model = model.to(args.device)
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images, masks = data_batch['image'].to(args.device), data_batch['mask'].to(args.device)
            outputs = model(images)
            loss = dice_loss_from_logits(outputs, masks)
            val_loss += loss.item()
            val_score += dice_score_from_logits(outputs, masks)

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_dice = val_score / len(val_loader)

    return val_epoch_loss, val_epoch_dice
