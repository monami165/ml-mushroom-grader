# Shiitake Mushroom Classification System

A computer vision system for classifying Shiitake mushrooms into 7 categories based on size, shape, and quality.

## Categories

1. **Baby Shiitake**: Cap diameter < 1 inch, round cap, no blemishes
2. **Small Shiitake**: Cap diameter 1-1.5 inches, round cap, no blemishes  
3. **Regular Shiitake**: Cap diameter 1.5-2 inches, round cap, no blemishes
4. **Large Shiitake**: Cap diameter 2.5-3 inches, round cap, no blemishes
5. **Extra Large Shiitake**: Cap diameter > 3 inches, round cap, no blemishes
6. **Shiitake B**: Cap diameter ≤ 1.5 inches, not round and/or blemished
7. **Sliced**: Cap diameter > 1.5 inches, not round and/or blemished

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your training data:
```
data/
├── Baby_Shiitake/
├── Small_Shiitake/
├── Regular_Shiitake/
├── Large_Shiitake/
├── Extra_Large_Shiitake/
├── Shiitake_B/
└── Sliced/
```

## Usage

### Training
```bash
python train_model.py --data_dir ./data --epochs 50 --batch_size 32
```

### Inference
```bash
python classify_mushroom.py --image path/to/mushroom.jpg --model mushroom_classifier.pth
```

## Model Features

- CNN architecture optimized for mushroom classification
- Data augmentation for robust training
- Special attention to size, shape, and blemish detection
- Tape measure reference integration for size estimation
