"""Utility functions for data processing and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from config import CLASS_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the Fashion MNIST dataset.
    
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Add channel dimension for CNN
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    return train_images, train_labels, test_images, test_labels


def plot_training_history(history: tf.keras.callbacks.History, save_path: str = None):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(model: tf.keras.Model, 
                           images: np.ndarray, 
                           labels: np.ndarray,
                           num_samples: int = 25,
                           save_path: str = None):
    """
    Plot sample predictions with images.
    
    Args:
        model: Trained Keras model
        images: Test images
        labels: True labels
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
    """
    # Get predictions
    predictions = model.predict(images[:num_samples], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate grid size
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        img = images[i].squeeze()
        ax.imshow(img, cmap='gray')
        
        # Get labels
        true_label = labels[i]
        pred_label = predicted_labels[i]
        
        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title
        ax.set_title(f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}", 
                    color=color, fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess a single image for prediction.
    
    Args:
        image: Input image (28x28 or 28x28x1)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure correct shape
    if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
        image = np.expand_dims(image, axis=-1)
    
    # Normalize
    if image.max() > 1.0:
        image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def get_class_name(class_idx: int) -> str:
    """
    Get class name from index.
    
    Args:
        class_idx: Class index (0-9)
        
    Returns:
        Class name string
    """
    return CLASS_NAMES[class_idx]
