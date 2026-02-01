"""Inference script for Fashion MNIST classifier."""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from config import CLASS_NAMES, MODELS_DIR
from utils import preprocess_image, get_class_name


class FashionMNISTPredictor:
    """Predictor class for Fashion MNIST classification."""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    def predict(self, image: np.ndarray, return_probabilities: bool = False):
        """
        Predict class for a single image.
        
        Args:
            image: Input image (28x28 or 28x28x1)
            return_probabilities: If True, return all class probabilities
            
        Returns:
            If return_probabilities=False: (predicted_class_idx, predicted_class_name, confidence)
            If return_probabilities=True: (predicted_class_idx, predicted_class_name, confidence, all_probabilities)
        """
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get predictions
        logits = self.model.predict(processed_image, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_name = get_class_name(predicted_idx)
        confidence = probabilities[predicted_idx]
        
        if return_probabilities:
            return predicted_idx, predicted_name, confidence, probabilities
        else:
            return predicted_idx, predicted_name, confidence
    
    def predict_batch(self, images: np.ndarray):
        """
        Predict classes for a batch of images.
        
        Args:
            images: Batch of images (N, 28, 28) or (N, 28, 28, 1)
            
        Returns:
            Tuple of (predicted_indices, predicted_names, confidences)
        """
        # Ensure correct shape
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Normalize if needed
        if images.max() > 1.0:
            images = images.astype('float32') / 255.0
        
        # Get predictions
        logits = self.model.predict(images, verbose=0)
        probabilities = tf.nn.softmax(logits).numpy()
        
        # Get predicted classes
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_names = [get_class_name(idx) for idx in predicted_indices]
        confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices]
        
        return predicted_indices, predicted_names, confidences
    
    def visualize_prediction(self, image: np.ndarray, true_label: int = None):
        """
        Visualize prediction with probabilities.
        
        Args:
            image: Input image (28x28 or 28x28x1)
            true_label: Optional true label for comparison
        """
        # Get prediction with probabilities
        pred_idx, pred_name, confidence, probabilities = self.predict(
            image, 
            return_probabilities=True
        )
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display image
        img = image.squeeze()
        ax1.imshow(img, cmap='gray')
        
        if true_label is not None:
            true_name = get_class_name(true_label)
            color = 'green' if pred_idx == true_label else 'red'
            ax1.set_title(f"True: {true_name}\nPredicted: {pred_name}\nConfidence: {confidence:.2%}",
                         color=color, fontsize=12, fontweight='bold')
        else:
            ax1.set_title(f"Predicted: {pred_name}\nConfidence: {confidence:.2%}",
                         fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Display probability distribution
        indices = np.arange(len(CLASS_NAMES))
        colors = ['green' if i == pred_idx else 'skyblue' for i in indices]
        
        ax2.barh(indices, probabilities, color=colors)
        ax2.set_yticks(indices)
        ax2.set_yticklabels(CLASS_NAMES)
        ax2.set_xlabel('Probability', fontsize=11)
        ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def predict_from_dataset(model_path: str, num_samples: int = 5):
    """
    Make predictions on random samples from the test dataset.
    
    Args:
        model_path: Path to the trained model
        num_samples: Number of samples to predict
    """
    # Load test data
    print("Loading test data...")
    (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize
    test_images = test_images.astype('float32') / 255.0
    
    # Create predictor
    predictor = FashionMNISTPredictor(model_path)
    
    # Get random samples
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    print(f"\nMaking predictions on {num_samples} random samples...\n")
    
    for i, idx in enumerate(indices, 1):
        image = test_images[idx]
        true_label = test_labels[idx]
        
        print(f"Sample {i}:")
        predictor.visualize_prediction(image, true_label)


def predict_from_file(model_path: str, image_path: str):
    """
    Make prediction on an image file.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image file
    """
    # Load image
    from PIL import Image
    
    print(f"Loading image from: {image_path}")
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    image = np.array(img).astype('float32') / 255.0
    
    # Create predictor
    predictor = FashionMNISTPredictor(model_path)
    
    # Make prediction
    print("\nMaking prediction...\n")
    predictor.visualize_prediction(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion MNIST Prediction")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for prediction"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test samples to predict (if --image not provided)"
    )
    
    args = parser.parse_args()
    
    if args.image:
        predict_from_file(args.model, args.image)
    else:
        predict_from_dataset(args.model, args.num_samples)
