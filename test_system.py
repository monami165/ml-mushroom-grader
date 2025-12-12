"""
Test script to validate the mushroom classification system
This script tests all components to ensure they work correctly
"""

import torch
import os
import tempfile
import json
import numpy as np
from PIL import Image
import argparse

# Import our modules
from data_utils import MushroomDataset, get_transforms, create_data_loaders, analyze_dataset
from model import get_model, count_parameters
from classify_mushroom import MushroomClassifier

def create_dummy_dataset(base_dir, num_images_per_class=5):
    """
    Create a small dummy dataset for testing
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
    
    print("Creating dummy dataset for testing...")
    
    # Create class directories
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create dummy images
        for i in range(num_images_per_class):
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = os.path.join(class_dir, f'dummy_image_{i}.jpg')
            img.save(img_path)
    
    print(f"Created dummy dataset with {len(classes)} classes and {num_images_per_class} images per class")
    return base_dir

def test_data_loading():
    """
    Test data loading functionality
    """
    print("\n" + "="*50)
    print("Testing Data Loading...")
    print("="*50)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        dummy_data_dir = create_dummy_dataset(temp_dir)
        
        # Test dataset analysis
        print("\n1. Testing dataset analysis...")
        total_images = analyze_dataset(dummy_data_dir)
        assert total_images > 0, "Dataset analysis failed"
        print("✅ Dataset analysis passed")
        
        # Test transforms
        print("\n2. Testing image transforms...")
        train_transform = get_transforms(input_size=224, is_training=True)
        val_transform = get_transforms(input_size=224, is_training=False)
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        train_tensor = train_transform(test_img)
        val_tensor = val_transform(test_img)
        
        assert train_tensor.shape == (3, 224, 224), f"Train transform failed: {train_tensor.shape}"
        assert val_tensor.shape == (3, 224, 224), f"Val transform failed: {val_tensor.shape}"
        print("✅ Image transforms passed")
        
        # Test dataset creation
        print("\n3. Testing dataset creation...")
        train_dataset = MushroomDataset(dummy_data_dir, transform=train_transform, split='train')
        val_dataset = MushroomDataset(dummy_data_dir, transform=val_transform, split='val')
        
        assert len(train_dataset) > 0, "Train dataset is empty"
        assert len(val_dataset) > 0, "Val dataset is empty"
        
        # Test data loading
        sample_image, sample_label = train_dataset[0]
        assert sample_image.shape == (3, 224, 224), f"Sample image shape wrong: {sample_image.shape}"
        assert isinstance(sample_label, int), "Sample label should be integer"
        print("✅ Dataset creation passed")
        
        # Test data loaders
        print("\n4. Testing data loaders...")
        train_loader, val_loader, class_names = create_data_loaders(
            dummy_data_dir, batch_size=4, input_size=224, num_workers=0
        )
        
        # Test one batch
        for batch_images, batch_labels in train_loader:
            assert batch_images.shape[0] <= 4, "Batch size too large"
            assert batch_images.shape[1:] == (3, 224, 224), f"Image shape wrong: {batch_images.shape}"
            assert len(batch_labels) == batch_images.shape[0], "Label count mismatch"
            break
        
        print("✅ Data loaders passed")
        print(f"Classes found: {class_names}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

def test_models():
    """
    Test model architectures
    """
    print("\n" + "="*50)
    print("Testing Model Architectures...")
    print("="*50)
    
    # Test ResNet model
    print("\n1. Testing ResNet-based model...")
    model_resnet = get_model(model_type='resnet', num_classes=7, pretrained=False)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model_resnet(dummy_input)
    
    assert output.shape == (2, 7), f"ResNet output shape wrong: {output.shape}"
    assert not torch.isnan(output).any(), "ResNet output contains NaN"
    
    param_count = count_parameters(model_resnet)
    print(f"✅ ResNet model passed - Parameters: {param_count:,}")
    
    # Test Lite model
    print("\n2. Testing Lite model...")
    model_lite = get_model(model_type='lite', num_classes=7)
    
    output = model_lite(dummy_input)
    
    assert output.shape == (2, 7), f"Lite output shape wrong: {output.shape}"
    assert not torch.isnan(output).any(), "Lite output contains NaN"
    
    param_count = count_parameters(model_lite)
    print(f"✅ Lite model passed - Parameters: {param_count:,}")
    
    # Test model saving/loading
    print("\n3. Testing model save/load...")
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Save model
        torch.save({
            'model_state_dict': model_resnet.state_dict(),
            'epoch': 0,
            'accuracy': 95.0,
            'loss': 0.1
        }, temp_path)
        
        # Load model
        checkpoint = torch.load(temp_path, map_location='cpu')
        model_loaded = get_model(model_type='resnet', num_classes=7, pretrained=False)
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        
        # Test loaded model
        output_loaded = model_loaded(dummy_input)
        assert torch.allclose(output, output_loaded, rtol=1e-5), "Loaded model output differs"
        print("✅ Model save/load passed")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_inference():
    """
    Test inference functionality
    """
    print("\n" + "="*50)
    print("Testing Inference System...")
    print("="*50)
    
    # Create a temporary model and image
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model
        model = get_model(model_type='resnet', num_classes=7, pretrained=False)
        model_path = os.path.join(temp_dir, 'test_model.pth')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 10,
            'accuracy': 95.0,
            'loss': 0.1
        }, model_path)
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        image_path = os.path.join(temp_dir, 'test_mushroom.jpg')
        test_image.save(image_path)
        
        print("\n1. Testing classifier initialization...")
        classifier = MushroomClassifier(model_path)
        print("✅ Classifier initialization passed")
        
        print("\n2. Testing single image prediction...")
        result = classifier.predict(image_path, return_probabilities=True)
        
        # Validate result structure
        required_keys = ['predicted_class', 'predicted_class_index', 'confidence', 'image_path', 'class_probabilities']
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"
        
        assert result['predicted_class'] in classifier.class_names, "Invalid predicted class"
        assert 0 <= result['confidence'] <= 1, "Invalid confidence value"
        assert len(result['class_probabilities']) == 7, "Wrong number of class probabilities"
        
        # Check probabilities sum to 1
        prob_sum = sum(result['class_probabilities'].values())
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities don't sum to 1: {prob_sum}"
        
        print("✅ Single image prediction passed")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        
        print("\n3. Testing batch prediction...")
        # Create multiple test images
        image_paths = []
        for i in range(3):
            test_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
            img_path = os.path.join(temp_dir, f'test_batch_{i}.jpg')
            test_img.save(img_path)
            image_paths.append(img_path)
        
        batch_results = classifier.predict_batch(image_paths, return_probabilities=True)
        
        assert len(batch_results) == 3, "Wrong number of batch results"
        for result in batch_results:
            assert 'predicted_class' in result, "Missing prediction in batch result"
            assert result['predicted_class'] in classifier.class_names, "Invalid batch prediction"
        
        print("✅ Batch prediction passed")
        
        print("\n4. Testing JSON output...")
        output_path = os.path.join(temp_dir, 'test_results.json')
        with open(output_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Verify JSON can be loaded
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert len(loaded_results) == len(batch_results), "JSON serialization failed"
        print("✅ JSON output passed")

def test_integration():
    """
    Test end-to-end integration
    """
    print("\n" + "="*50)
    print("Testing End-to-End Integration...")
    print("="*50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal dataset
        dummy_data_dir = create_dummy_dataset(temp_dir, num_images_per_class=2)
        
        print("\n1. Testing training data pipeline...")
        train_loader, val_loader, class_names = create_data_loaders(
            dummy_data_dir, batch_size=2, input_size=224, num_workers=0
        )
        
        # Test one training step
        model = get_model(model_type='lite', num_classes=7, pretrained=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss), "Training loss is NaN"
            print(f"  Training step completed - Loss: {loss.item():.4f}")
            break
        
        print("✅ Training pipeline passed")
        
        print("\n2. Testing model persistence...")
        model_path = os.path.join(temp_dir, 'integration_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 1,
            'accuracy': 85.0,
            'loss': loss.item()
        }, model_path)
        
        # Test inference with saved model
        classifier = MushroomClassifier(model_path)
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        image_path = os.path.join(temp_dir, 'integration_test.jpg')
        test_image.save(image_path)
        
        result = classifier.predict(image_path)
        assert result['predicted_class'] in class_names, "Integration prediction failed"
        
        print("✅ End-to-end integration passed")

def main():
    parser = argparse.ArgumentParser(description='Test Mushroom Classification System')
    parser.add_argument('--skip-data', action='store_true', help='Skip data loading tests')
    parser.add_argument('--skip-models', action='store_true', help='Skip model tests')
    parser.add_argument('--skip-inference', action='store_true', help='Skip inference tests')
    parser.add_argument('--skip-integration', action='store_true', help='Skip integration tests')
    
    args = parser.parse_args()
    
    print("🧪 Starting Mushroom Classification System Tests")
    print("="*60)
    
    try:
        if not args.skip_data:
            test_data_loading()
        
        if not args.skip_models:
            test_models()
        
        if not args.skip_inference:
            test_inference()
        
        if not args.skip_integration:
            test_integration()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ The mushroom classification system is ready to use")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Organize your mushroom images in the data/ directory")
        print("2. Run: python train_model.py --data_dir ./data")
        print("3. Use: python classify_mushroom.py --image your_mushroom.jpg")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
