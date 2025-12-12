import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

class MushroomDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.transform = transform
        
        # Define class labels mapping
        self.classes = [
            'Baby_Shiitake',
            'Small_Shiitake', 
            'Regular_Shiitake',
            'Large_Shiitake',
            'Extra_Large_Shiitake',
            'Shiitake_B',
            'Sliced'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        all_images = []
        all_labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_dir, img_name)
                        all_images.append(img_path)
                        all_labels.append(self.class_to_idx[class_name])
        
        # Split data into train and validation sets
        if len(all_images) > 0:
            train_images, val_images, train_labels, val_labels = train_test_split(
                all_images, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
            )
            
            if split == 'train':
                self.images = train_images
                self.labels = train_labels
            else:
                self.images = val_images
                self.labels = val_labels
        else:
            self.images = []
            self.labels = []
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(input_size=224, is_training=True):
    """
    Get image transforms with augmentation focused on mushroom classification
    Special attention to preserving size relationships and shape information
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),  # Slightly larger for random crop
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Limited rotation to preserve shape
            transforms.ColorJitter(
                brightness=0.2,      # Moderate brightness variation
                contrast=0.2,        # Contrast changes to help detect blemishes
                saturation=0.15,     # Slight saturation changes for hue variation
                hue=0.1             # Small hue changes to detect blemishes
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Help with edge detection
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, batch_size=32, input_size=224, num_workers=4):
    """
    Create train and validation data loaders
    """
    # Get transforms
    train_transform = get_transforms(input_size, is_training=True)
    val_transform = get_transforms(input_size, is_training=False)
    
    # Create datasets
    train_dataset = MushroomDataset(data_dir, transform=train_transform, split='train')
    val_dataset = MushroomDataset(data_dir, transform=val_transform, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def analyze_dataset(data_dir):
    """
    Analyze the dataset and print statistics
    """
    classes = [
        'Baby_Shiitake',
        'Small_Shiitake', 
        'Regular_Shiitake',
        'Large_Shiitake',
        'Extra_Large_Shiitake',
        'Shiitake_B',
        'Sliced'
    ]
    
    print("Dataset Analysis:")
    print("-" * 50)
    
    total_images = 0
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            print(f"{class_name}: {count} images")
            total_images += count
        else:
            print(f"{class_name}: 0 images (directory not found)")
    
    print("-" * 50)
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(classes)}")
    
    return total_images
