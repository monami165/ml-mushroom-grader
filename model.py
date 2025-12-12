import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MushroomClassifier(nn.Module):
    """
    CNN model optimized for mushroom classification with attention to:
    - Cap size (through spatial attention)
    - Shape detection (through edge-preserving convolutions)
    - Blemish detection (through texture analysis)
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(MushroomClassifier, self).__init__()
        
        # Use ResNet18 as backbone for feature extraction
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        
        # Get feature dimension (512 for ResNet18)
        backbone_out_features = 512
        
        # Spatial Attention Module for size/shape focus
        self.spatial_attention = SpatialAttentionModule()
        
        # Channel Attention Module for texture/blemish focus
        self.channel_attention = ChannelAttentionModule(backbone_out_features)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)  # Shape: [batch, 512, 7, 7]
        
        # Apply spatial attention (focuses on mushroom cap area)
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        # Apply channel attention (focuses on texture/color channels for blemishes)
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        # Global pooling
        features = self.global_avg_pool(features)  # Shape: [batch, 512, 1, 1]
        features = features.view(features.size(0), -1)  # Shape: [batch, 512]
        
        # Classification
        output = self.classifier(features)
        
        return output

class SpatialAttentionModule(nn.Module):
    """
    Spatial attention to focus on mushroom cap area
    Helps with size and shape detection
    """
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute spatial statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max across channels
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_input)
        attention_map = self.sigmoid(attention_map)
        
        return attention_map

class ChannelAttentionModule(nn.Module):
    """
    Channel attention to focus on texture and color features
    Helps with blemish detection
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(batch_size, channels, 1, 1)
        
        return attention

class MushroomClassifierLite(nn.Module):
    """
    Lighter version of the mushroom classifier for faster training/inference
    """
    def __init__(self, num_classes=7):
        super(MushroomClassifierLite, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_model(model_type='resnet', num_classes=7, pretrained=True):
    """
    Factory function to get the appropriate model
    """
    if model_type == 'resnet':
        return MushroomClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'lite':
        return MushroomClassifierLite(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
