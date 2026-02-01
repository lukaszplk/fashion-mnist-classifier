"""Training script for Fashion MNIST classifier."""

import argparse
import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from config import (
    MODEL_PATH,
    MODELS_DIR,
    LOGS_DIR,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR
)
from model import get_model
from utils import (
    load_and_preprocess_data,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    print_classification_report
)
import numpy as np


def get_callbacks(model_save_path: str):
    """
    Create training callbacks.
    
    Args:
        model_save_path: Path to save the best model
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save the best model
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def train_model(model_type: str = "cnn", 
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE,
                save_visualizations: bool = True):
    """
    Train the Fashion MNIST classifier.
    
    Args:
        model_type: Type of model - "simple" or "cnn"
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_visualizations: Whether to save visualization plots
    """
    print("="*70)
    print("FASHION MNIST CLASSIFIER - TRAINING")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    
    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    model = get_model(model_type)
    model.summary()
    
    # Setup callbacks
    model_save_path = MODELS_DIR / f"best_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    callbacks = get_callbacks(str(model_save_path))
    
    # Train model
    print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")
    print(f"Validation split: {VALIDATION_SPLIT*100}%")
    
    history = model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy, test_sparse_acc, test_top3_acc = model.evaluate(
        test_images, 
        test_labels, 
        verbose=1
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Top-3 Accuracy: {test_top3_acc:.4f} ({test_top3_acc*100:.2f}%)")
    
    # Get predictions for confusion matrix
    print("\nGenerating predictions for evaluation...")
    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Print classification report
    print_classification_report(test_labels, predicted_labels)
    
    # Save visualizations
    if save_visualizations:
        print("\nGenerating visualizations...")
        viz_dir = LOGS_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Training history
        plot_training_history(
            history, 
            save_path=str(viz_dir / f"training_history_{timestamp}.png")
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            test_labels, 
            predicted_labels,
            save_path=str(viz_dir / f"confusion_matrix_{timestamp}.png")
        )
        
        # Sample predictions
        plot_sample_predictions(
            model,
            test_images,
            test_labels,
            num_samples=25,
            save_path=str(viz_dir / f"sample_predictions_{timestamp}.png")
        )
    
    # Save final model
    final_model_path = MODELS_DIR / f"final_model_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {model_save_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fashion MNIST Classifier")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn",
        choices=["simple", "cnn"],
        help="Type of model to train (default: cnn)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable saving visualization plots"
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_visualizations=not args.no_viz
    )
