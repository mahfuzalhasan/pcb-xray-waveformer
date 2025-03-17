import os
import json
import shutil
import numpy as np

import torch
# Helper function to save split annotations
def save_split_annotations(file_names, coco_data, output_path):
    # Filter images
    images = [img for img in coco_data['images'] if img['file_name'] in file_names]
    image_ids = [img['id'] for img in images]

    # Filter annotations
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids]

    # Create new COCO data dictionary
    new_coco_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': images,
        'annotations': annotations,
        'categories': coco_data['categories']
    }

    # Save new annotations
    with open(output_path, 'w') as f:
        json.dump(new_coco_data, f, indent=4)

# Helper function to copy images to their respective folders
def copy_images(file_names, source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for file_name in file_names:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))

# Set up the optimizer with Layer-wise Learning Rate Decay
def get_optimizer_with_llrd(model, base_lr=1e-4, lr_decay_factor=0.95):
    seen_params = set()
    optimizer_grouped_parameters = []
    for i, layer in enumerate(model.sam_model.vision_encoder.layers):
        layer_params = [p for n, p in layer.named_parameters() if p not in seen_params]
        seen_params.update(layer_params)
        optimizer_grouped_parameters.append({'params': layer_params, 'lr': base_lr * (lr_decay_factor ** (len(model.sam_model.vision_encoder.layers) - i))})

    adapter_params = [p for n, p in model.named_parameters() if "adapter" in n and p not in seen_params]
    seen_params.update(adapter_params)
    optimizer_grouped_parameters.append({'params': adapter_params, 'lr': base_lr})

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    return optimizer

# Dice score function
def dice_score(pred, target, smooth=1):
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)