import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from model import get_model

class MushroomClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize the mushroom classifier
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Computing device (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Class names for Shiitake mushroom categories
        self.class_names = [
            'Baby_Shiitake',
            'Small_Shiitake',
            'Regular_Shiitake', 
            'Large_Shiitake',
            'Extra_Large_Shiitake',
            'Shiitake_B',
            'Sliced'
        ]
        
        # Create and load model
        self.model = get_model(model_type='resnet', num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best validation accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor, image, original_size
        
        except Exception as e:
            raise RuntimeError(f"Error preprocessing image: {e}")
    
    def predict(self, image_path, return_probabilities=False):
        """
        Classify a mushroom image
        
        Args:
            image_path: Path to the mushroom image
            return_probabilities: If True, return class probabilities
        
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            # Get all class probabilities
            all_probs = probabilities[0].cpu().numpy()
        
        # Prepare results
        results = {
            'predicted_class': self.class_names[predicted_class],
            'predicted_class_index': predicted_class,
            'confidence': confidence,
            'image_path': image_path,
            'image_size': original_size
        }
        
        if return_probabilities:
            results['class_probabilities'] = {
                self.class_names[i]: float(prob) for i, prob in enumerate(all_probs)
            }
        
        return results
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Classify multiple mushroom images
        
        Args:
            image_paths: List of paths to mushroom images
            return_probabilities: If True, return class probabilities
        
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': None
                })
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None, show_probabilities=True):
        """
        Visualize prediction with annotated image
        
        Args:
            image_path: Path to the input image
            save_path: Path to save the visualization (optional)
            show_probabilities: Whether to show class probabilities
        """
        # Get prediction
        results = self.predict(image_path, return_probabilities=show_probabilities)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        if show_probabilities and 'class_probabilities' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # Display image with prediction
        ax1.imshow(image)
        ax1.axis('off')
        
        # Add prediction text
        prediction_text = f"Prediction: {results['predicted_class']}\n"
        prediction_text += f"Confidence: {results['confidence']:.3f}"
        ax1.set_title(prediction_text, fontsize=14, fontweight='bold')
        
        # Show class probabilities bar chart
        if show_probabilities and 'class_probabilities' in results:
            probs = results['class_probabilities']
            classes = list(probs.keys())
            values = list(probs.values())
            
            # Create bar chart
            bars = ax2.bar(range(len(classes)), values)
            ax2.set_xlabel('Mushroom Categories')
            ax2.set_ylabel('Probability')
            ax2.set_title('Class Probabilities')
            ax2.set_xticks(range(len(classes)))
            ax2.set_xticklabels(classes, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            
            # Highlight the predicted class
            predicted_idx = results['predicted_class_index']
            bars[predicted_idx].set_color('red')
            bars[predicted_idx].set_alpha(0.8)
            
            # Add probability values on bars
            for i, v in enumerate(values):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return results

def get_category_description(class_name):
    """
    Get detailed description of mushroom category
    """
    descriptions = {
        'Baby_Shiitake': 'Cap diameter < 1 inch, round cap, no blemishes',
        'Small_Shiitake': 'Cap diameter 1-1.5 inches, round cap, no blemishes',
        'Regular_Shiitake': 'Cap diameter 1.5-2 inches, round cap, no blemishes',
        'Large_Shiitake': 'Cap diameter 2.5-3 inches, round cap, no blemishes',
        'Extra_Large_Shiitake': 'Cap diameter > 3 inches, round cap, no blemishes',
        'Shiitake_B': 'Cap diameter ≤ 1.5 inches, not round and/or blemished',
        'Sliced': 'Cap diameter > 1.5 inches, not round and/or blemished'
    }
    return descriptions.get(class_name, 'Unknown category')

def main():
    parser = argparse.ArgumentParser(description='Classify Shiitake Mushroom Images')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to mushroom image to classify')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization of prediction')
    parser.add_argument('--save_viz', type=str, default=None,
                        help='Path to save visualization image')
    parser.add_argument('--probabilities', action='store_true',
                        help='Show class probabilities')
    parser.add_argument('--batch', type=str, nargs='+', default=None,
                        help='Process multiple images')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Initialize classifier
        classifier = MushroomClassifier(args.model)
        
        if args.batch:
            # Batch processing
            print(f"Processing {len(args.batch)} images...")
            results = classifier.predict_batch(args.batch, return_probabilities=args.probabilities)
            
            # Print results
            for result in results:
                if 'error' in result:
                    print(f"\n❌ {result['image_path']}: {result['error']}")
                else:
                    print(f"\n📸 Image: {result['image_path']}")
                    print(f"🍄 Prediction: {result['predicted_class']}")
                    print(f"📊 Confidence: {result['confidence']:.3f}")
                    print(f"📝 Description: {get_category_description(result['predicted_class'])}")
                    
                    if args.probabilities and 'class_probabilities' in result:
                        print("📈 Class Probabilities:")
                        for class_name, prob in result['class_probabilities'].items():
                            print(f"   {class_name}: {prob:.3f}")
        
        else:
            # Single image processing
            if not os.path.exists(args.image):
                print(f"❌ Image not found: {args.image}")
                return
            
            print(f"🔍 Analyzing mushroom image: {args.image}")
            
            if args.visualize:
                results = classifier.visualize_prediction(
                    args.image, 
                    save_path=args.save_viz,
                    show_probabilities=args.probabilities
                )
            else:
                results = classifier.predict(args.image, return_probabilities=args.probabilities)
            
            # Print detailed results
            print(f"\n🍄 Classification Results:")
            print(f"📸 Image: {results['image_path']}")
            print(f"📏 Image Size: {results['image_size'][0]}x{results['image_size'][1]} pixels")
            print(f"🎯 Predicted Class: {results['predicted_class']}")
            print(f"📊 Confidence: {results['confidence']:.3f}")
            print(f"📝 Description: {get_category_description(results['predicted_class'])}")
            
            if args.probabilities and 'class_probabilities' in results:
                print(f"\n📈 All Class Probabilities:")
                sorted_probs = sorted(results['class_probabilities'].items(), key=lambda x: x[1], reverse=True)
                for class_name, prob in sorted_probs:
                    indicator = "👑" if class_name == results['predicted_class'] else "  "
                    print(f"{indicator} {class_name}: {prob:.3f}")
            
            results = [results]  # Make it a list for consistent output format
        
        # Save results to JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
