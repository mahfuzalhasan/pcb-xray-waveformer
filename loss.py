import torch
import torch.nn as nn
# Dice Loss Function for Multiple Classes
class DiceLoss(nn.Module):
    def __init__(self, num_classes=5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.softmax(inputs, dim=1)  # Use softmax for multi-class segmentation
        loss = 0
        for c in range(self.num_classes):
            input_c = inputs[:, c, :, :]
            target_c = (targets == c).float()
            intersection = (input_c * target_c).sum()
            dice = (2. * intersection + smooth) / (input_c.sum() + target_c.sum() + smooth)
            loss += 1 - dice
        return loss / self.num_classes