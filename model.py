import torch
import torch.nn as nn


class SpatialAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAdapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class SAMWithSpatialAdapters(nn.Module):
    def __init__(self, sam_model, adapter_channels=256, num_classes=5):  # Add num_classes parameter
        super(SAMWithSpatialAdapters, self).__init__()
        self.sam_model = sam_model
        # Initialize SpatialAdapter with 256 input and output channels
        self.adapters = nn.ModuleList([SpatialAdapter(256, adapter_channels) for _ in range(sam_model.config.vision_config.num_hidden_layers)])
        self.final_conv = nn.Conv2d(adapter_channels, num_classes, kernel_size=1)  # Change to num_classes

    def forward(self, pixel_values):
        vision_outputs = self.sam_model.vision_encoder(pixel_values)
        x = vision_outputs[0]  # Assuming vision_outputs[0] is the feature map
        # print(f"vision_outputs[0] shape: {x.shape}")  # Debug print to inspect shape
        for adapter in self.adapters:
            x = adapter(x)
        x = self.final_conv(x)  # Reduce to num_classes channels
        # print(f"Final output shape before upsampling: {x.shape}")  # Debug print to inspect shape
        return x