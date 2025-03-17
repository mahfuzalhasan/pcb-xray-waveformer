import cv2
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from pycocotools import mask as coco_mask
# Custom dataset class
class PCBXRayDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.image_info = self.coco_data['images']
        self.annotations = {ann['image_id']: ann for ann in self.coco_data['annotations']}
        self.category_info = self.coco_data['categories']

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        image_id = img_info['id']
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        anns = [ann for ann in self.annotations.values() if ann['image_id'] == image_id]
        for ann in anns:
            rle = coco_mask.frPyObjects(ann['segmentation'], image.shape[0], image.shape[1])
            decoded_mask = coco_mask.decode(rle)
            if len(decoded_mask.shape) == 3:
                decoded_mask = decoded_mask[:, :, 0]  # Take the first channel if it's a multi-channel mask
            mask += np.squeeze(decoded_mask).astype(np.uint8)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask