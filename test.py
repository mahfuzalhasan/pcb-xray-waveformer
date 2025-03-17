import numpy as np
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from loss import DiceLoss
from utils import dice_score
from transformers import SamModel, SamProcessor, SamConfig
from model import SAMWithSpatialAdapters
from dataset_xray import PCBXRayDataset
from transformation import CustomTransforms

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(model, data_loader, device, num_classes=5):
    model.eval()
    dice_scores = []
    precision_list = []
    recall_list = []
    test_loss = 0.0

    criterion = DiceLoss(num_classes=num_classes)

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            # Ensure outputs and masks have the correct types
            outputs = outputs.float()
            masks = masks.long()
            
            outputs_softmax = torch.softmax(outputs, dim=1)  # Use softmax for multi-class segmentation

            for c in range(num_classes):
                pred_mask = (torch.argmax(outputs_softmax, dim=1) == c).float()
                true_mask = (masks == c).float()
                dice = dice_score(pred_mask.cpu().numpy(), true_mask.cpu().numpy())
                dice_scores.append(dice)

                precision = precision_score(true_mask.cpu().numpy().flatten(), pred_mask.cpu().numpy().flatten(), zero_division=0)
                recall = recall_score(true_mask.cpu().numpy().flatten(), pred_mask.cpu().numpy().flatten(), zero_division=0)

                precision_list.append(precision)
                recall_list.append(recall)

            loss = criterion(outputs, masks)
            test_loss += loss.item()

    mean_dice = np.mean(dice_scores)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    test_loss = test_loss / len(data_loader)

    return test_loss, mean_dice, mean_precision, mean_recall

best_model_path = "saved_best_model/best_model_epoch_25.pth"
test_image_folder_path = "test/"
test_annotation_path = "annotation/test211121516.json"

# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sam_model = SamModel.from_pretrained('facebook/sam-vit-huge')
# Load the best model for evaluation on the test set
best_model = SAMWithSpatialAdapters(sam_model, num_classes=5).to(device)
best_model.load_state_dict(torch.load(best_model_path))

# Load the test dataset
test_dataset = PCBXRayDataset(test_image_folder_path, test_annotation_path, transforms=CustomTransforms(is_train=False))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Evaluate the best model on the test dataset
test_loss, test_dice, test_precision, test_recall = evaluate_model_on_test_set(best_model, test_loader, num_classes=5)

print(f"Test Loss: {test_loss}")
print(f"Test Dice Score: {test_dice}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")