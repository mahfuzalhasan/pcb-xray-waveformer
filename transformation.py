import torch
import torch.nn as nn
from torchvision import transforms
# Define data augmentation and normalization transforms
class CustomTransforms:
    def __init__(self, is_train=True):
        if is_train:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, image, mask):
        image = self.transforms(image)
        mask = Image.fromarray(mask)
        mask = F.resize(mask, (1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)
        mask = transforms.ToTensor()(mask).long().squeeze(0)  # Ensure mask is 2D
        return image, mask