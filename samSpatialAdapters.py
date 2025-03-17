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
from utils import get_optimizer_with_llrd, dice_score
from loss import DiceLoss
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

