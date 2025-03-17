import json
import os
import random
import shutil
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from transformers import SamModel, SamProcessor, SamConfig
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score


from model import SAMWithSpatialAdapters
from dataset_xray import PCBXRayDataset
from utils import get_optimizer_with_llrd, dice_score, save_split_annotations, copy_images
from loss import DiceLoss
from transformation import CustomTransforms
# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
root_path = "/home/UFAD/mdmahfuzalhasan/Documents/data/Sam_PEFT"
image_folder_path = f"{root_path}/Images/"
annotation_path = f"{root_path}/annotation/pcb1to15.json"
output_base_path = f"{root_path}/model_results/resultsEpoch25_withoutCP/"

# Load the annotations
with open(annotation_path, 'r') as f:
    coco_data = json.load(f)

# Get all image file names
image_files = [img['file_name'] for img in coco_data['images']]

# Load the pre-trained SAM model
sam_model = SamModel.from_pretrained('facebook/sam-vit-huge')

# Integrate spatial adapters
model_with_adapters = SAMWithSpatialAdapters(sam_model).to(device)
optimizer = get_optimizer_with_llrd(model_with_adapters)

# Training function
def train_model(train_loader, val_loader, fold, fold_output_path, num_epochs=25, num_classes=5):
    model = SAMWithSpatialAdapters(sam_model, num_classes=num_classes).to(device)
    criterion = DiceLoss(num_classes=num_classes)
    optimizer = get_optimizer_with_llrd(model)

    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            # Debug prints
            #print(f"Epoch {epoch + 1}, Fold {fold + 1}, outputs shape: {outputs.shape}, masks shape: {masks.shape}")

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                # Debug prints
                #print(f"Epoch {epoch + 1}, Fold {fold + 1}, outputs shape: {outputs.shape}, masks shape: {masks.shape}")

                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(fold_output_path, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)

    return train_loss / len(train_loader), val_loss / len(val_loader), best_model_path

# Evaluation function
def evaluate_model(model, data_loader, num_classes=5):
    model.eval()
    dice_scores = []
    precision_list = []
    recall_list = []

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            outputs = torch.argmax(outputs, dim=1)  # Get the class with the highest score

            for c in range(num_classes):
                pred_mask = (outputs == c).float()
                true_mask = (masks == c).float()
                dice = dice_score(pred_mask.cpu().numpy(), true_mask.cpu().numpy())
                dice_scores.append(dice)

                precision = precision_score(true_mask.cpu().numpy().flatten(), pred_mask.cpu().numpy().flatten(), zero_division=0)
                recall = recall_score(true_mask.cpu().numpy().flatten(), pred_mask.cpu().numpy().flatten(), zero_division=0)

                precision_list.append(precision)
                recall_list.append(recall)

    mean_dice = np.mean(dice_scores)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)

    return mean_dice, mean_precision, mean_recall


# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
best_model_dice = {'score': 0, 'path': None}
best_model_precision = {'score': 0, 'path': None}

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
    print(f'Fold {fold + 1}')
    train_files = [image_files[i] for i in train_idx]
    val_files = [image_files[i] for i in val_idx]

    fold_output_path = os.path.join(output_base_path, f'fold_{fold + 1}')
    train_image_path = os.path.join(fold_output_path, 'train')
    val_image_path = os.path.join(fold_output_path, 'val')

    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(val_image_path, exist_ok=True)

    save_split_annotations(train_files, coco_data, os.path.join(fold_output_path, 'train_annotations.json'))
    save_split_annotations(val_files, coco_data, os.path.join(fold_output_path, 'val_annotations.json'))
    copy_images(train_files, image_folder_path, train_image_path)
    copy_images(val_files, image_folder_path, val_image_path)

    train_dataset = PCBXRayDataset(train_image_path, os.path.join(fold_output_path, 'train_annotations.json'), transforms=CustomTransforms(is_train=True))
    val_dataset = PCBXRayDataset(val_image_path, os.path.join(fold_output_path, 'val_annotations.json'), transforms=CustomTransforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_loss, val_loss, best_model_path = train_model(train_loader, val_loader, fold, num_classes=5)

    # Evaluate on validation set
    model_with_adapters.load_state_dict(torch.load(best_model_path))
    mean_dice, mean_precision, mean_recall = evaluate_model(model_with_adapters, val_loader, num_classes=5)

    fold_results.append({
        'fold': fold + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_model_path': best_model_path,
        'mean_dice': mean_dice,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
    })

    if mean_dice > best_model_dice['score']:
        best_model_dice['score'] = mean_dice
        best_model_dice['path'] = best_model_path

    if mean_precision > best_model_precision['score']:
        best_model_precision['score'] = mean_precision
        best_model_precision['path'] = best_model_path

# Calculate average results across all folds
avg_train_loss = np.mean([result['train_loss'] for result in fold_results])
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
avg_mean_dice = np.mean([result['mean_dice'] for result in fold_results])
avg_mean_precision = np.mean([result['mean_precision'] for result in fold_results])
avg_mean_recall = np.mean([result['mean_recall'] for result in fold_results])

# Output the final results
print("Cross-validation results:", fold_results)
print(f"Average Train Loss: {avg_train_loss}")
print(f"Average Val Loss: {avg_val_loss}")
print(f"Average Dice Score: {avg_mean_dice}")
print(f"Average Precision: {avg_mean_precision}")
print(f"Average Recall: {avg_mean_recall}")

# Output the best models
print(f"Best model based on Dice Score: {best_model_dice['path']} with score {best_model_dice['score']}")
print(f"Best model based on Precision: {best_model_precision['path']} with score {best_model_precision['score']}")




