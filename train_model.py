import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
from tqdm import tqdm
import json

from data_utils import create_data_loaders, analyze_dataset
from model import get_model, count_parameters

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()
        
        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct_predictions / total_samples
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device, class_names):
    """
    Validate the model
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct_predictions / total_samples
    
    # Print per-class accuracies
    print("\nPer-class accuracies:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'{class_name}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        else:
            print(f'{class_name}: No samples')
    
    return epoch_loss, epoch_acc

def save_model(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, filepath)

def save_training_plots(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Save training plots
    """
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Shiitake Mushroom Classifier')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'lite'],
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Analyze dataset
    total_images = analyze_dataset(args.data_dir)
    if total_images == 0:
        print("Error: No images found in dataset directory!")
        print("Please organize your data as follows:")
        print("data/")
        print("├── Baby_Shiitake/")
        print("├── Small_Shiitake/") 
        print("├── Regular_Shiitake/")
        print("├── Large_Shiitake/")
        print("├── Extra_Large_Shiitake/")
        print("├── Shiitake_B/")
        print("└── Sliced/")
        return
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Classes: {class_names}")
    
    # Create model
    model = get_model(model_type=args.model_type, num_classes=len(class_names), pretrained=True)
    model = model.to(device)
    
    print(f"\nModel: {args.model_type}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['accuracy']
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, class_names)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{args.epochs} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_model(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_model(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        print("=" * 60)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    save_model(model, optimizer, args.epochs - 1, val_loss, val_acc, final_model_path)
    
    # Save training plots
    save_training_plots(train_losses, val_losses, train_accs, val_accs, args.save_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'class_names': class_names
    }
    
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models and plots saved in: {args.save_dir}")
    print(f"Use the best_model.pth for inference")

if __name__ == '__main__':
    main()
