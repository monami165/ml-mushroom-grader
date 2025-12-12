# Shiitake Mushroom Classification - Usage Guide

This guide provides detailed instructions for training and using the mushroom classification system.

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv mushroom_env
mushroom_env\Scripts\activate  # Windows
# source mushroom_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your 50-80 photos per category in the following structure:

```
data/
├── Baby_Shiitake/          # Cap < 1", round, no blemishes
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Small_Shiitake/         # Cap 1-1.5", round, no blemishes
│   ├── image1.jpg
│   └── ...
├── Regular_Shiitake/       # Cap 1.5-2", round, no blemishes
├── Large_Shiitake/         # Cap 2.5-3", round, no blemishes
├── Extra_Large_Shiitake/   # Cap > 3", round, no blemishes
├── Shiitake_B/            # Cap ≤ 1.5", not round/blemished
└── Sliced/                # Cap > 1.5", not round/blemished
```

### 3. Test Your Setup

```bash
python test_system.py
```

### 4. Train the Model

```bash
python train_model.py --data_dir ./data --epochs 50 --batch_size 32
```

### 5. Classify Mushrooms

```bash
python classify_mushroom.py --image path/to/mushroom.jpg --model ./checkpoints/best_model.pth --visualize
```

## Detailed Instructions

### Training Configuration

The training script offers many customization options:

```bash
python train_model.py \
    --data_dir ./data \
    --model_type resnet \          # or 'lite' for faster training
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --input_size 224 \
    --save_dir ./checkpoints
```

**Key Parameters:**
- `--model_type`: Choose 'resnet' (higher accuracy) or 'lite' (faster training)
- `--epochs`: More epochs = better training (50-100 recommended)
- `--batch_size`: Reduce if out of memory, increase if you have more GPU memory
- `--learning_rate`: Default 0.001 works well, try 0.0001 for fine-tuning

### Model Selection

**ResNet Model (Recommended):**
- Higher accuracy (~95%+)
- Uses pre-trained ImageNet weights
- Better for production use
- ~11M parameters

**Lite Model:**
- Faster training and inference
- Good for quick testing
- ~2M parameters
- Accuracy ~90%+

### Training Tips

1. **Data Quality**: Ensure photos clearly show mushroom caps with tape measures for size reference
2. **Balanced Dataset**: Try to have similar numbers of images per category
3. **Image Quality**: Use clear, well-lit photos with minimal background clutter
4. **Training Time**: Expect 1-3 hours for 50 epochs (depending on hardware)

### Monitoring Training

The training script provides:
- Real-time progress bars
- Per-epoch accuracy and loss
- Validation metrics
- Training plots saved as PNG
- Automatic best model saving

### Classification Usage

**Basic Classification:**
```bash
python classify_mushroom.py --image mushroom.jpg --model checkpoints/best_model.pth
```

**With Visualization:**
```bash
python classify_mushroom.py \
    --image mushroom.jpg \
    --model checkpoints/best_model.pth \
    --visualize \
    --probabilities \
    --save_viz result_visualization.png
```

**Batch Processing:**
```bash
python classify_mushroom.py \
    --batch image1.jpg image2.jpg image3.jpg \
    --model checkpoints/best_model.pth \
    --output results.json
```

## Understanding Results

### Classification Categories

1. **Baby Shiitake**: Small, perfect specimens (< 1 inch diameter)
2. **Small Shiitake**: Medium-small, perfect specimens (1-1.5 inches)
3. **Regular Shiitake**: Standard size, perfect specimens (1.5-2 inches)
4. **Large Shiitake**: Large, perfect specimens (2.5-3 inches)
5. **Extra Large Shiitake**: Very large, perfect specimens (> 3 inches)
6. **Shiitake B**: Small imperfect specimens (≤ 1.5 inches, not round/blemished)
7. **Sliced**: Large imperfect specimens (> 1.5 inches, not round/blemished)

### Confidence Scores

- **> 0.9**: Very confident prediction
- **0.7-0.9**: Good confidence
- **0.5-0.7**: Moderate confidence (review recommended)
- **< 0.5**: Low confidence (may need manual inspection)

## Troubleshooting

### Common Issues

**1. "No images found in dataset directory"**
- Check folder structure matches exactly
- Ensure image files are .jpg, .jpeg, .png, .bmp, or .tiff
- Verify folder names have no typos

**2. "CUDA out of memory"**
- Reduce batch size: `--batch_size 16` or `--batch_size 8`
- Use lite model: `--model_type lite`
- Reduce input size: `--input_size 192`

**3. "Low validation accuracy"**
- Increase training epochs: `--epochs 100`
- Check data quality and labeling
- Ensure balanced dataset
- Try different learning rate: `--learning_rate 0.0001`

**4. "Model file not found"**
- Check the model path is correct
- Ensure training completed successfully
- Look for `best_model.pth` in checkpoints directory

### Performance Optimization

**For Better Accuracy:**
- Use more training data (100+ images per class)
- Train for more epochs (100-200)
- Use data augmentation (built-in)
- Ensure high-quality, consistent photos

**For Faster Training:**
- Use lite model
- Reduce input size to 192 or 160
- Increase batch size if you have GPU memory
- Use fewer epochs for initial testing

## Advanced Usage

### Custom Data Augmentation

Edit `data_utils.py` to modify augmentation:

```python
transforms.ColorJitter(
    brightness=0.3,      # Increase for more brightness variation
    contrast=0.3,        # Increase for more contrast variation
    saturation=0.2,      # Adjust saturation changes
    hue=0.15            # Adjust hue changes for blemish detection
)
```

### Fine-tuning Pre-trained Model

```bash
python train_model.py \
    --data_dir ./data \
    --resume ./checkpoints/best_model.pth \
    --learning_rate 0.0001 \
    --epochs 20
```

### Batch Processing Script

Create a simple batch processor:

```python
from classify_mushroom import MushroomClassifier
import os

classifier = MushroomClassifier('./checkpoints/best_model.pth')
image_dir = './test_images'
results = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        result = classifier.predict(image_path)
        results.append({
            'filename': filename,
            'prediction': result['predicted_class'],
            'confidence': result['confidence']
        })
        print(f"{filename}: {result['predicted_class']} ({result['confidence']:.3f})")
```

## Support

If you encounter issues:
1. Run the test system: `python test_system.py`
2. Check this guide for common solutions
3. Verify your data organization
4. Ensure all dependencies are installed correctly
