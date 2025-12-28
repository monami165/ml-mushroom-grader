#!/usr/bin/env python3
"""
Example setup and usage script for the Shiitake Mushroom Classification System
This script demonstrates how to organize data and run the system
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the recommended directory structure for the project"""
    
    directories = [
        'data/Baby_Shiitake',
        'data/Small_Shiitake', 
        'data/Regular_Shiitake',
        'data/Large_Shiitake',
        'data/Extra_Large_Shiitake',
        'data/Shiitake_B',
        'data/Sliced',
        'checkpoints',
        'results',
        'test_images'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    print("\nDirectory structure created successfully!")
    print("\nNext steps:")
    print("1. Place your mushroom photos in the appropriate data/ subdirectories")
    print("2. Ensure each photo shows mushrooms with a tape measure for size reference")
    print("3. Aim for 50-80 photos per category for best results")
    
    return directories

def create_example_config():
    """Create example configuration files"""
    
    # Training configuration example
    train_config = """# Example training command
python train_model.py \\
    --data_dir ./data \\
    --model_type resnet \\
    --epochs 50 \\
    --batch_size 32 \\
    --learning_rate 0.001 \\
    --input_size 224 \\
    --save_dir ./checkpoints

# For faster training (less accurate):
# python train_model.py --data_dir ./data --model_type lite --epochs 30 --batch_size 16
"""
    
    # Inference configuration example  
    inference_config = """# Example inference commands

# Single image classification
python classify_mushroom.py \\
    --image test_images/mushroom1.jpg \\
    --model checkpoints/best_model.pth \\
    --visualize \\
    --probabilities

# Batch processing
python classify_mushroom.py \\
    --batch test_images/mushroom1.jpg test_images/mushroom2.jpg test_images/mushroom3.jpg \\
    --model checkpoints/best_model.pth \\
    --output results/batch_results.json

# Save visualization
python classify_mushroom.py \\
    --image test_images/mushroom1.jpg \\
    --model checkpoints/best_model.pth \\
    --visualize \\
    --save_viz results/classification_result.png
"""
    
    with open('example_training.sh', 'w') as f:
        f.write(train_config)
    
    with open('example_inference.sh', 'w') as f:
        f.write(inference_config)
    
    print("Created example configuration files:")
    print("  ✓ example_training.sh - Training examples")
    print("  ✓ example_inference.sh - Inference examples")

def print_data_organization_guide():
    """Print detailed guide for organizing mushroom photos"""
    
    print("\n" + "="*60)
    print("MUSHROOM PHOTO ORGANIZATION GUIDE")
    print("="*60)
    
    categories = {
        'Baby_Shiitake': 'Cap diameter < 1 inch, round cap, no blemishes',
        'Small_Shiitake': 'Cap diameter 1-1.5 inches, round cap, no blemishes',
        'Regular_Shiitake': 'Cap diameter 1.5-2 inches, round cap, no blemishes', 
        'Large_Shiitake': 'Cap diameter 2.5-3 inches, round cap, no blemishes',
        'Extra_Large_Shiitake': 'Cap diameter > 3 inches, round cap, no blemishes',
        'Shiitake_B': 'Cap diameter ≤ 1.5 inches, not round and/or blemished',
        'Sliced': 'Cap diameter > 1.5 inches, not round and/or blemished'
    }
    
    print("\nPlace your mushroom photos in these folders based on the criteria:")
    print()
    
    for i, (folder, description) in enumerate(categories.items(), 1):
        print(f"{i}. data/{folder}/")
        print(f"   → {description}")
        print(f"   → Place 50-80 photos here")
        print()
    
    print("PHOTO REQUIREMENTS:")
    print("• Clear, well-lit images showing the mushroom cap clearly")
    print("• Include a tape measure in each photo for size reference")
    print("• Minimal background clutter")
    print("• Consistent lighting and angle when possible")
    print("• Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
    print()
    
    print("NAMING SUGGESTIONS:")
    print("• baby_shiitake_001.jpg, baby_shiitake_002.jpg, etc.")
    print("• Use descriptive names to help with organization")
    print("• Avoid special characters and spaces in filenames")

def main():
    print("🍄 Shiitake Mushroom Classification System Setup")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("⚠️  Warning: requirements.txt not found.")
        print("Make sure you're running this script from the project root directory.")
        print()
    
    # Create directories
    create_directory_structure()
    print()
    
    # Create example configs
    create_example_config()
    print()
    
    # Print organization guide
    print_data_organization_guide()
    
    print("="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print()
    print("🚀 Quick Start:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test system: python test_system.py") 
    print("3. Organize your photos in the data/ folders")
    print("4. Train model: python train_model.py --data_dir ./data")
    print("5. Classify: python classify_mushroom.py --image your_image.jpg --model checkpoints/best_model.pth")
    print()
    print("📖 For detailed instructions, see: USAGE_GUIDE.md")
    print("🧪 To validate your setup, run: python test_system.py")

if __name__ == '__main__':
    main()
